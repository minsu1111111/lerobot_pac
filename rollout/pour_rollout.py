"""물 따르기 데모 롤아웃 루프 (lerobot-rollout 을 대신함 — 그쪽 CLI 는 감지기·TTS 를 끼울 곳이 없다).

[lerobot v0.6.1 의 관측 → 정책 → 행동 흐름 (소스 확인) 과 이 스크립트의 대응]
  lerobot-rollout (scripts/lerobot_rollout.py → rollout/context.py build_rollout_context)
    1. 정책: PreTrainedConfig → get_policy_class(type).from_pretrained(path, config) → .to(device).eval()
             make_pre_post_processors(cfg, pretrained_path, preprocessor_overrides={"device_processor": device})
    2. 로봇: make_robot_from_config(BiSOFollowerConfig) → connect() → 초기 관절값 저장 (종료 때 복귀용)
    3. 특징: hw_to_dataset_features(robot.observation_features, "observation")
             → "observation.state"(관절 12, robot 순서) + "observation.images.{top,left_wrist,right_wrist}"
    4. 루프 (strategies/base.py BaseStrategy.run, 매 tick):
         obs = robot.get_observation()                       # 관절 float + 카메라 uint8 HWC
         obs = robot_observation_processor(obs)              # 기본값 = 항등
         frame = build_dataset_frame(features, obs, "observation")   # state 벡터 + 이미지
         SyncInferenceEngine.get_action(frame):
           prepare_observation_for_inference  (torch, 이미지 /255·CHW, 배치, device, task, robot_type)
           → preprocessor (정규화) → policy.select_action (ACT: 큐가 비면 청크 100개 추론, 아니면 pop)
           → postprocessor (역정규화, cpu) → make_robot_action → action_keys 순서 텐서
         robot_action_processor((action, obs)) (항등) → robot.send_action(action)
           → BiSOFollower 가 left_/right_ 로 나눠 SOFollower.send_action
             (max_relative_target 이 있으면 현재 위치 기준으로 자르고 Goal_Position 씀)
         precise_sleep(1/fps - dt), 넘치면 경고
    5. 종료: return_to_initial_position(기본 True, 3s 보간) → disconnect (disable_torque_on_disconnect 기본 True)
  이 스크립트는 1·3·4 를 같은 함수로 그대로 따라 하고(정책 추론 경로 동일), 그 사이에
  상태 기계·완료 감지기·TTS·키 정지·명령 클램프·로그를 끼운다.

[상태 기계]
  IDLE   : TTS ready → 명령 대기. 기본 --stt-mode vad = Enter 없이 말소리 자동 감지 ("물 따라줘"),
           --stt-mode enter = Enter 로 녹음 시작/끝. 어느 쪽이든 p/s/q + Enter 키보드 입력도 받음.
           안내 음성이 끝난 뒤에만 듣는다 ("물을 따르겠습니다" 의 '따르' 를 스스로 명령으로 듣지 않게).
  POUR   : pour 명령 → TTS start → policy/processor/감지기 리셋 → 30fps 제어 루프
           감지기(pour_detector)는 '관측된' 오른팔 wrist_roll(state[10]) 을 본다
           DONE → 즉시 TTS done → --post-done-s(기본 12s, 물통 내려놓기 ≈10s 중앙/15s 최대) 동안 계속 → 정지
           --timeout-s(기본 thresholds.json timeout_suggest_s=63) 넘으면 TTS timeout → 정지 (DONE 전일 때만)
  정지는 '행동 전송 중단' = 모터는 마지막 목표 위치를 유지 (토크 유지). 다시 IDLE.
  수동 정지: 붓는 동안 Space 또는 s → 즉시 정지 + TTS stopped, q → 정지 후 프로그램 종료,
            Ctrl+C → 안전 종료 (real 은 finally 에서 초기자세 복귀 후 disconnect).
  붓는 동안 STT 는 꺼져 있다 (마이크 대기는 IDLE 에서만). 정지는 키보드로만.

[청크 경계 점프]  n_action_steps=100(학습값)이면 청크가 바뀌는 프레임에 명령이 한 번에 크게 튄다.
  완화책: --n-action-steps 30/50, --temporal-ensemble 0.01 (ACT 규칙상 n_action_steps=1 강제, 매 프레임 추론),
         --max-step-deg (직전 명령 대비 프레임당 변화 제한, 그리퍼는 --max-step-gripper).
  기본값: --max-step-deg 10 (시연 데이터의 프레임당 최대 변화 ≈10°; 정상 동작에선 몇 프레임만 걸림). 0 이면 끔.
  실제 로봇은 추가로 --max-relative-target (lerobot 의 현재 위치 기준 제한) 를 줄 수 있다.

실행 예:
  python pour_rollout.py --backend replay --episode 0 --auto-start --no-stt        # 실시간 재생 시험
  python pour_rollout.py --backend replay --episode 0 --auto-start --no-stt --fast # 잠 안 자고 최대 속도 (점프 측정용)
  python pour_rollout.py --backend replay+mujoco --episode 3 --auto-start --no-stt
  python pour_rollout.py --backend real --robot-id bimanual --left-port /dev/follower1 ...   (run_demo.sh 참고)
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import JOINT_KEYS  # noqa: E402
from keys import RawKeys  # noqa: E402
from paths import DATASET, GITHUB, MODEL_REPO, OUTPUTS, THRESHOLDS  # noqa: E402
from pour_detector import DONE, PourDetector, PourDetectorConfig  # noqa: E402

FPS = 30
ROLL = JOINT_KEYS.index("right_wrist_roll.pos")  # 10
GRIP = [JOINT_KEYS.index("left_gripper.pos"), JOINT_KEYS.index("right_gripper.pos")]
ARM = [i for i in range(12) if i not in GRIP]
TASK = "Pick up the cup with the left arm and pour water from the bottle into it with the right arm"
# rollout/thresholds.json 이 없을 때의 기본값 = 2026-10 joint_analysis 결과
DEFAULT_TH = dict(joint="right_wrist_roll.pos", sign=1.0, tilt_off=78.88, return_off=39.44, hold=5,
                  base_frames=30, timeout_suggest_s=63.0)


# --------------------------------------------------------------------------- #
# 주변 모듈 (다른 담당). 없으면 대체 동작.
# --------------------------------------------------------------------------- #
class PrintTTS:
    """tts_player 가 없을 때: 자막만 출력."""

    def __init__(self):
        try:
            sys.path.insert(0, str(GITHUB / "tts"))
            from phrases import PHRASES
        except Exception:
            PHRASES = {}
        self.phrases = PHRASES

    def say(self, key):
        text = self.phrases.get(key, key)
        print(f"[TTS(자막만)] {text}", flush=True)
        return text

    def wait(self, timeout=None):
        return True

    def close(self):
        pass


def make_tts(enabled: bool):
    try:
        sys.path.insert(0, str(GITHUB / "tts"))
        from tts_player import TTSPlayer

        return TTSPlayer(wav_dir=GITHUB / "tts" / "wavs", enabled=enabled)
    except Exception as e:  # 모듈 없음/wav 없음 등
        print(f"[TTS] tts_player 사용 불가 → 자막만 출력 ({type(e).__name__}: {e})")
        return PrintTTS()


class KeyboardCommander:
    """voice_command 가 없거나 --no-stt 일 때: 줄 입력 (p=붓기, s=정지, q=종료)."""

    def listen(self):
        from types import SimpleNamespace

        try:
            line = input("\n[대기] p+Enter = 붓기 시작 | q+Enter = 종료 > ").strip().lower()
        except EOFError:
            line = "q"
        cmd = {"p": "pour", "": None, "s": "stop", "q": "quit"}.get(line[:1] if line else "", None)
        return SimpleNamespace(text=line, command=cmd, source="keyboard", duration_s=0.0, stt_s=0.0)


def make_commander(use_stt: bool, model: str, mic):
    if use_stt:
        try:
            sys.path.insert(0, str(GITHUB / "stt"))
            from voice_command import VoiceCommander

            return VoiceCommander(model=model, device="auto", mic=mic)
        except Exception as e:
            print(f"[STT] voice_command 사용 불가 → 키보드 입력 ({type(e).__name__}: {e})")
    return KeyboardCommander()


# --------------------------------------------------------------------------- #
# 정책 (SyncInferenceEngine 과 같은 경로)
# --------------------------------------------------------------------------- #
def resolve_model_path(model: str) -> str:
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser())
    from huggingface_hub import snapshot_download

    try:  # 캐시 우선 (현장 오프라인)
        return snapshot_download(model, local_files_only=True)
    except Exception:
        return snapshot_download(model)


class Policy:
    def __init__(self, model: str, device: str, n_action_steps: int | None, temporal_ensemble: float | None,
                 task: str = TASK, keep_backbone_download: bool = False, sync_exact: bool = False):
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self.torch = torch
        path = resolve_model_path(model)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device, cfg.pretrained_path = device, path
        self.trained_n_action_steps = cfg.n_action_steps
        if temporal_ensemble is not None:
            # configuration_act.py __post_init__: temporal_ensemble_coeff 를 쓰면 n_action_steps 는 1 이어야 함
            cfg.temporal_ensemble_coeff, cfg.n_action_steps = float(temporal_ensemble), 1
        elif n_action_steps is not None:
            if not 1 <= n_action_steps <= cfg.chunk_size:
                raise ValueError(f"n_action_steps 는 1..{cfg.chunk_size}")
            cfg.n_action_steps = n_action_steps
        if not keep_backbone_download:
            # ACT 는 생성 시 torchvision ResNet18 ImageNet 가중치를 받는다(~/.cache/torch/hub). 체크포인트가
            # backbone 까지 전부 덮어쓰므로 필요 없다 → None 으로 꺼서 오프라인 다운로드 위험 제거.
            # (tools/offline_check.py 가 두 방식의 출력이 같음을 확인)
            cfg.pretrained_backbone_weights = None
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.device = torch.device(device)
        self.task = task
        self.path = path
        # SyncInferenceEngine 은 매 tick 전처리를 돈다. 같은 결과를 더 싸게: ACT 큐 pop 때는 생략
        self.skip_pre_on_pop = (not sync_exact and cfg.type == "act" and cfg.temporal_ensemble_coeff is None)

        from lerobot.utils.feature_utils import hw_to_dataset_features

        from backends import _features

        # lerobot-rollout 과 같은 방식으로 robot 특징 → dataset 특징 (BiSOFollower 와 같은 키·순서)
        feats = _features()
        self.features = {**hw_to_dataset_features({k: float for k in JOINT_KEYS}, "action"),
                         **hw_to_dataset_features(feats, "observation")}
        exp = {k for k, v in cfg.input_features.items()}
        got = {k for k in self.features if k.startswith("observation")}
        assert exp <= got, f"정책 입력 {exp} ⊄ 로봇 특징 {got}"

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def will_infer(self) -> bool:
        """이번 select_action 에서 새 청크를 추론하는지 (ACT 큐가 비었거나 temporal ensemble)."""
        q = getattr(self.policy, "_action_queue", None)
        return q is None or len(q) == 0

    def __call__(self, obs: dict) -> np.ndarray:
        torch = self.torch
        from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
        from lerobot.utils.feature_utils import build_dataset_frame

        amp = (torch.autocast(device_type="cuda") if self.device.type == "cuda" and self.cfg.use_amp
               else nullcontext())
        with torch.inference_mode(), amp:
            if self.skip_pre_on_pop and not self.will_infer():
                # ACT select_action 은 큐가 차 있으면 batch 를 보지 않고 popleft 만 한다 (modeling_act.py).
                # 그래서 이미지 전처리(3캠 float 변환·정규화, CPU 10ms+)를 건너뛴다. 결과는 동일.
                a = self.policy.select_action(None)
            else:
                frame = build_dataset_frame(self.features, obs, prefix="observation")
                x = prepare_observation_for_inference(frame, self.device, self.task, "bi_so_follower")
                x = self.pre(x)
                a = self.policy.select_action(x)
            a = self.post(a)
        ad = make_robot_action(a.squeeze(0).cpu(), self.features)
        return np.array([ad[k] for k in JOINT_KEYS], dtype=np.float32)

    def warmup(self, n=2):
        """첫 추론(CUDA 초기화 등)이 붓기 첫 프레임을 막지 않도록 미리 한 번 돌린다."""
        from backends import CAM_SHAPES

        obs = {k: 0.0 for k in JOINT_KEYS}
        obs.update({c: np.zeros(s, np.uint8) for c, s in CAM_SHAPES.items()})
        ts = []
        for _ in range(n):
            self.reset()
            t = time.perf_counter()
            self(obs)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        self.reset()
        return ts


# --------------------------------------------------------------------------- #
# 한 번의 붓기
# --------------------------------------------------------------------------- #
def clamp_step(cmd, ref, max_deg, max_grip):
    """직전 명령(ref) 대비 프레임당 변화량 제한. 몸통 관절은 도, 그리퍼는 0~100 단위."""
    lim = np.full(12, np.inf, np.float32)
    if max_deg:
        lim[ARM] = max_deg
    if max_grip:
        lim[GRIP] = max_grip
    return ref + np.clip(cmd - ref, -lim, lim)


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return dict(n=int(len(x)), mean=float(x.mean()), p50=float(np.median(x)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def run_pour(args, robot, policy, tts, det_cfg, run_dir: Path, run_idx: int):
    """제어 루프 1회. (요약 dict, quit 여부) 반환. 행 로그는 메모리에 모았다가 끝나고 CSV 로 쓴다."""
    from lerobot.utils.robot_utils import precise_sleep

    period = 1.0 / FPS
    policy.reset()
    det = PourDetector(det_cfg)  # 관측 roll → 실제 판단
    det_cmd = PourDetector(det_cfg)  # 명령 roll → 참고용 (replay 에서 모델이 붓는 동작을 냈는지)
    if hasattr(robot, "restart"):
        robot.restart()

    rows, events_all = [], []
    prev_sent = None
    stop_reason, quit_req = None, False
    done_t = None
    last_status = -1.0
    warn_n, last_warn = 0, -10.0
    t_start = time.perf_counter()
    step = 0

    def ev(name, t):
        events_all.append((round(t, 3), name))
        return name

    subtitle = {"text": "", "t": -1e9}  # 시뮬 영상 자막용

    def say(key, t):
        print()  # 상태줄 다음 줄에
        subtitle.update(text=tts.say(key) or key, t=t)
        return ev(f"tts:{key}", t)

    with RawKeys() as keys:
        if not keys.enabled:
            print("[키] stdin 이 터미널이 아님 → Space/s/q 정지 비활성 (Ctrl+C 만 가능)")

        def _loop():
            nonlocal stop_reason, quit_req, done_t, last_status, warn_n, last_warn, step, prev_sent
            evs = [say("start", 0.0)]
            while True:
                loop_t0 = time.perf_counter()
                t = step / FPS if args.fast else loop_t0 - t_start

                # ---- 수동 정지 (키) ----
                k = keys.poll() or ""
                if args.test_stop_at is not None and t >= args.test_stop_at:
                    k += "s"  # 시험용: 정지 키를 누른 것처럼
                if "q" in k or "Q" in k:
                    stop_reason, quit_req = "quit", True
                elif " " in k or "s" in k or "S" in k:
                    stop_reason = "manual"
                if stop_reason:
                    evs.append(say("stopped", t))
                    evs.append(ev(f"stop:{stop_reason}", t))
                    rows.append(dict(step=step, t=t, event=";".join(evs)))
                    break

                # ---- 관측 → 정책 → 행동 ----
                obs = robot.get_observation()
                t_obs = time.perf_counter()
                state = np.array([obs[k] for k in JOINT_KEYS], dtype=np.float32)
                new_chunk = policy.will_infer()
                cmd = policy(obs)
                t_inf = time.perf_counter()
                ref = prev_sent if prev_sent is not None else state
                sent = clamp_step(cmd, ref, args.max_step_deg, args.max_step_gripper)
                robot.send_action({k: float(v) for k, v in zip(JOINT_KEYS, sent)})
                t_act = time.perf_counter()
                jump = np.abs(sent - prev_sent) if prev_sent is not None else np.zeros(12, np.float32)
                prev_sent = sent

                # ---- 완료 감지 ----
                e = det.update(float(state[ROLL]))
                if e:
                    evs.append(ev(f"det:{e}", t))
                    if e == DONE:
                        done_t = t
                        evs.append(say("done", t))  # 팀 결정: DONE 즉시 안내, 동작은 post-done 동안 계속
                e2 = det_cmd.update(float(cmd[ROLL]))
                if e2:
                    evs.append(ev(f"det_cmd:{e2}", t))

                if hasattr(robot, "set_overlay"):  # replay+mujoco 영상에 상태·자막 표시
                    lines = [f"{t:5.1f}s   감지 {det.state}   오른손목 roll {state[ROLL]:6.1f}°"]
                    if done_t is not None:
                        lines.append(f"완료 후 {t - done_t:4.1f}s / 정지 {args.post_done_s:.0f}s")
                    if t - subtitle["t"] < 4.0:
                        lines.append(f"[TTS] {subtitle['text']}")
                    robot.set_overlay("\n".join(lines))

                work = t_act - loop_t0
                row = dict(step=step, t=t, work_ms=work * 1e3, obs_ms=(t_obs - loop_t0) * 1e3,
                           infer_ms=(t_inf - t_obs) * 1e3, new_chunk=int(new_chunk), det=det.state,
                           det_cmd=det_cmd.state, roll_obs=float(state[ROLL]), roll_cmd=float(cmd[ROLL]),
                           roll_sent=float(sent[ROLL]), jump_arm=float(jump[ARM].max()),
                           jump_roll=float(jump[ROLL]), jump_grip=float(jump[GRIP].max()),
                           clamped=int(np.any(np.abs(sent - cmd) > 1e-4)))
                row.update({f"obs_{i}": float(v) for i, v in enumerate(state)})
                row.update({f"cmd_{i}": float(v) for i, v in enumerate(sent)})

                # ---- 종료 조건 ----
                if done_t is not None and t - done_t >= args.post_done_s:
                    stop_reason = "done"
                elif done_t is None and t >= args.timeout_s:
                    evs.append(say("timeout", t))
                    stop_reason = "timeout"
                elif getattr(robot, "finished", False):
                    stop_reason = "replay_end"
                if stop_reason:
                    evs.append(ev(f"stop:{stop_reason}", t))
                row["event"] = ";".join(evs)
                evs = []

                # ---- 상태줄 / 주기 ----
                if t - last_status >= (0.5 if sys.stdout.isatty() else 5.0):
                    last_status = t
                    line = (f"[{t:6.1f}s] det={det.state:7s} roll obs {state[ROLL]:7.1f} cmd {cmd[ROLL]:7.1f}  "
                            f"work {work * 1e3:5.1f}ms infer {(t_inf - t_obs) * 1e3:6.1f}ms  jump {jump[ARM].max():5.1f}")
                    if done_t is not None:
                        line += f"  정지까지 {args.post_done_s - (t - done_t):4.1f}s"
                    print(line, end="\r" if sys.stdout.isatty() else "\n", flush=True)
                if stop_reason:
                    rows.append(row)
                    break
                if not args.fast:
                    dt = time.perf_counter() - loop_t0
                    if dt < period:
                        precise_sleep(period - dt)
                    else:
                        warn_n += 1
                        if t - last_warn > 5.0:
                            last_warn = t
                            print(f"\n[경고] 루프 지연 {dt * 1e3:.0f}ms > {period * 1e3:.0f}ms "
                                  f"(추론 {(t_inf - t_obs) * 1e3:.0f}ms, 누적 {warn_n}회)")
                row["period_ms"] = (time.perf_counter() - loop_t0) * 1e3
                rows.append(row)
                step += 1

        try:
            _loop()
        except KeyboardInterrupt:  # 행동 전송을 즉시 멈추고 로그는 저장한 뒤 프로그램 종료
            stop_reason, quit_req = "ctrl_c", True
            tts.say("stopped")
            events_all.append((round(time.perf_counter() - t_start, 3), "stop:ctrl_c"))
    print()

    # ---- 저장 / 요약 ----
    csv_path = run_dir / f"run_{run_idx:02d}.csv"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    R = [r for r in rows if "work_ms" in r]
    jumps_b = [r["jump_arm"] for r in R[1:] if r["new_chunk"]]
    jumps_i = [r["jump_arm"] for r in R[1:] if not r["new_chunk"]]
    summ = dict(
        run=run_idx, csv=str(csv_path), stop_reason=stop_reason, steps=len(R),
        duration_s=(R[-1]["t"] if R else 0.0),
        wall_s=time.perf_counter() - t_start,
        done_t=done_t, done_reason=det.done_reason, tilt_t=(det.tilt_step / FPS if det.tilt_step is not None else None),
        cmd_detector=dict(state=det_cmd.state, done_t=(det_cmd.done_step / FPS if det_cmd.done_step is not None else None),
                          tilt_t=(det_cmd.tilt_step / FPS if det_cmd.tilt_step is not None else None)),
        baseline=det.baseline,
        period_ms=stats([r["period_ms"] for r in R if "period_ms" in r]),
        work_ms=stats([r["work_ms"] for r in R]),
        infer_ms_chunk=stats([r["infer_ms"] for r in R if r["new_chunk"]]),
        infer_ms_pop=stats([r["infer_ms"] for r in R if not r["new_chunk"]]),
        overruns=int(sum(r["work_ms"] > period * 1e3 for r in R)),
        jump_arm_boundary=stats(jumps_b), jump_arm_within=stats(jumps_i),
        jump_arm_all=stats([r["jump_arm"] for r in R[1:]]),
        jump_roll_all=stats([r["jump_roll"] for r in R[1:]]),
        jump_grip_all=stats([r["jump_grip"] for r in R[1:]]),
        clamped_steps=int(sum(r["clamped"] for r in R)),
        events=events_all,
    )
    if hasattr(robot, "decode_wait_s"):
        summ["replay_decode_wait_s"] = robot.decode_wait_s
    if hasattr(robot, "state") and hasattr(robot, "episode"):  # 시연 자체의 DONE (참고)
        d = PourDetector(det_cfg).run(robot.state[:, ROLL])
        summ["demo_done_t"] = d.done_step / FPS if d.state == DONE else None
        summ["episode"], summ["episode_len_s"] = robot.episode, robot.length / FPS
    print(f"[결과] 정지={stop_reason}  DONE={done_t if done_t is None else round(done_t, 2)}s  "
          f"steps={len(R)}  지연 {summ['overruns']}회  "
          f"점프(팔, 최대) 경계 {summ['jump_arm_boundary']['max'] if jumps_b else 0:.1f} / 내부 "
          f"{summ['jump_arm_within']['max'] if jumps_i else 0:.1f}  → {csv_path}")
    return summ, quit_req


# --------------------------------------------------------------------------- #
def load_thresholds(path: Path) -> dict:
    if path.is_file():
        return {**DEFAULT_TH, **json.loads(path.read_text())}
    print(f"[감지기] {path} 없음 → 내장 기본값 사용")
    return dict(DEFAULT_TH)


def make_robot(args):
    from backends import DatasetReplayRobot, RealRobot, ReplayMujocoRobot, install_calib

    clock = "step" if args.fast else args.replay_clock
    if args.backend == "replay":
        return DatasetReplayRobot(args.episode, Path(args.dataset), clock=clock)
    if args.backend == "replay+mujoco":
        return ReplayMujocoRobot(args.episode, Path(args.dataset), clock=clock, render=args.sim_render,
                                 video_path=args.sim_video)
    if args.calib_left or args.calib_right:
        install_calib(args.robot_id, args.calib_left, args.calib_right)
    need = dict(left_port=args.left_port, right_port=args.right_port, top_cam=args.top_cam,
                left_cam=args.left_cam, right_cam=args.right_cam)
    missing = [k for k, v in need.items() if not v]
    if missing:
        raise SystemExit(f"real 백엔드 인자 부족: {missing} (run_demo.sh / robot.env 참고)")
    return RealRobot(args.robot_id, max_relative_target=args.max_relative_target,
                     keep_torque=args.keep_torque, fourcc=args.fourcc, **need)


def parse_args(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_argument_group("백엔드")
    g.add_argument("--backend", choices=["real", "replay", "replay+mujoco"], default="replay")
    g.add_argument("--episode", type=int, default=0, help="replay 에피소드 번호")
    g.add_argument("--dataset", default=str(DATASET))
    g.add_argument("--replay-clock", choices=["step", "wall"], default="step")
    g.add_argument("--fast", action="store_true", help="잠 안 자고 최대 속도 (시간 = step/30, replay 전용 측정용)")
    g.add_argument("--sim-render", choices=["none", "window", "offscreen"], default="none")
    g.add_argument("--sim-video", default=None, help="replay+mujoco 영상 저장 경로 (offscreen)")
    g = ap.add_argument_group("real 로봇")
    g.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "bimanual"),
                   help="BiSOFollower id → 캘리브레이션 {id}_left.json / {id}_right.json")
    g.add_argument("--left-port", default=os.environ.get("LEFT_PORT"))
    g.add_argument("--right-port", default=os.environ.get("RIGHT_PORT"))
    g.add_argument("--top-cam", default=os.environ.get("TOP_CAM"))
    g.add_argument("--left-cam", default=os.environ.get("LEFT_CAM"))
    g.add_argument("--right-cam", default=os.environ.get("RIGHT_CAM"))
    g.add_argument("--fourcc", default=os.environ.get("CAM_FOURCC") or None)
    g.add_argument("--calib-left", default=os.environ.get("CALIB_LEFT"), help="예: follower1 (복사 원본)")
    g.add_argument("--calib-right", default=os.environ.get("CALIB_RIGHT"), help="예: follower2")
    g.add_argument("--max-relative-target", type=float, default=None,
                   help="lerobot 안전 제한: 현재 위치 대비 목표 최대 차이 (도). 기본 없음(학습 때와 같음)")
    g.add_argument("--keep-torque", action="store_true", help="종료 때 토크 유지 (기본: lerobot 처럼 토크 끔)")
    g.add_argument("--no-return-home", action="store_true", help="종료 때 초기 자세 복귀 생략")
    g = ap.add_argument_group("정책")
    g.add_argument("--policy", default=MODEL_REPO, help="HF repo id 또는 로컬 폴더")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--n-action-steps", type=int, default=None, help="청크에서 실제 쓰는 개수 (학습값 100)")
    g.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEF",
                   help="ACT temporal ensembling (원 논문 0.01). n_action_steps=1, 매 프레임 추론")
    g.add_argument("--max-step-deg", type=float, default=10.0,
                   help="몸통 관절 프레임당 명령 변화 제한 (도). 기본 10, 0 이면 끔")
    g.add_argument("--max-step-gripper", type=float, default=None, help="그리퍼 프레임당 명령 변화 제한 (0~100)")
    g.add_argument("--task", default=TASK)
    g.add_argument("--threads", type=int, default=None, help="torch CPU 스레드 수 (기본: torch 기본값)")
    g.add_argument("--sync-exact", action="store_true",
                   help="SyncInferenceEngine 처럼 매 프레임 전처리 (기본: 큐 pop 프레임은 생략, 결과 동일)")
    g = ap.add_argument_group("데모 흐름")
    g.add_argument("--thresholds", default=str(THRESHOLDS))
    g.add_argument("--post-done-s", type=float, default=12.0)
    g.add_argument("--timeout-s", type=float, default=None, help="기본: thresholds.json timeout_suggest_s")
    g.add_argument("--no-stt", action="store_true")
    g.add_argument("--no-tts", action="store_true")
    g.add_argument("--stt-model", default="small")
    g.add_argument("--stt-mode", choices=("vad", "enter"), default="vad",
                   help="vad = Enter 없이 말소리 자동 감지 (기본), enter = Enter 로 녹음 시작/끝")
    g.add_argument("--vad-level", type=int, default=2, choices=range(4), help="VAD 민감도 0(관대)~3(엄격). 시끄러우면 높임")
    g.add_argument("--silence-ms", type=int, default=1000, help="vad: 이 시간 무음이면 말 끝으로 봄")
    g.add_argument("--mic", default=None)
    g.add_argument("--auto-start", action="store_true", help="명령 없이 바로 1회 붓고 종료 (시험용)")
    g.add_argument("--test-stop-at", type=float, default=None, help="시험용: 이 시각(s)에 정지 키 입력 흉내")
    g.add_argument("--out", default=str(OUTPUTS / "rollout"))
    g.add_argument("--tag", default="")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    th = load_thresholds(Path(args.thresholds))
    assert th["joint"].endswith("right_wrist_roll.pos"), th["joint"]
    if args.timeout_s is None:
        args.timeout_s = float(th["timeout_suggest_s"])
    det_cfg = PourDetectorConfig(tilt_off=th["tilt_off"], return_off=th["return_off"], hold=th["hold"],
                                 base_frames=th["base_frames"], sign=th["sign"])
    if args.fast and args.backend == "real":
        raise SystemExit("--fast 는 replay 전용")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / (stamp + (f"_{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    print(f"[정책] {args.policy} 로딩 (device={args.device}) ...")
    policy = Policy(args.policy, args.device, args.n_action_steps, args.temporal_ensemble, args.task,
                    sync_exact=args.sync_exact)
    cfg = policy.cfg
    print(f"[정책] chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"temporal_ensemble={cfg.temporal_ensemble_coeff} ← {policy.path}")
    if cfg.n_action_steps >= 100 and cfg.temporal_ensemble_coeff is None and not args.max_step_deg:
        print("[경고] n_action_steps=100 (학습값): 청크 경계마다 명령이 한 프레임에 크게 튈 수 있음 "
              "(replay open-loop 측정: 팔 관절 최대 18~70°/프레임, 시연은 ≤10°). --n-action-steps / --temporal-ensemble / --max-step-deg 참고")
    wt = policy.warmup()
    print(f"[정책] 워밍업 추론 {', '.join(f'{x * 1e3:.0f}ms' for x in wt)}")
    if cfg.temporal_ensemble_coeff is not None and wt[-1] > 1 / FPS and not args.fast:
        print(f"[경고] temporal ensemble 은 매 프레임 추론 ({wt[-1] * 1e3:.0f}ms > 33ms) → 30fps 불가")

    robot = make_robot(args)
    tts = make_tts(not args.no_tts)
    commander = None if args.auto_start else make_commander(not args.no_stt, args.stt_model, args.mic)
    session = dict(args=vars(args), thresholds=th, started=stamp, device=args.device,
                   policy_path=policy.path, n_action_steps=cfg.n_action_steps,
                   temporal_ensemble=cfg.temporal_ensemble_coeff, warmup_ms=[x * 1e3 for x in wt], runs=[],
                   idle_events=[])

    def save():
        (run_dir / "summary.json").write_text(json.dumps(session, indent=2, ensure_ascii=False, default=str))

    try:
        print(f"[로봇] {args.backend} 연결 ...")
        robot.connect()
        run_idx, announce = 0, True
        if args.stt_mode == "vad" and hasattr(commander, "listen_auto"):
            def listen():
                return commander.listen_auto(vad_level=args.vad_level, silence_ms=args.silence_ms)
        else:
            listen = getattr(commander, "listen", None)  # --auto-start 면 commander 없음
        while True:
            # ---- IDLE ----
            if args.auto_start:
                if run_idx >= 1:
                    break
            else:
                if announce:
                    tts.say("ready")
                    announce = False
                tts.wait(timeout=8.0)  # 안내 음성이 마이크에 들어가지 않도록
                time.sleep(0.3)  # 스피커 잔향
                r = listen()
                session["idle_events"].append(dict(t=time.time(), text=r.text, command=r.command,
                                                   source=r.source, stt_s=r.stt_s))
                if r.command == "quit":
                    break
                if r.command != "pour":
                    # 자동 감지는 소음에도 켜질 수 있어, 받아쓴 말이 있을 때만 다시 말해 달라고 한다
                    if r.command is None and (args.stt_mode == "enter" or (r.text or "").strip()):
                        tts.say("retry")
                    continue
            # ---- POUR ----
            run_idx += 1
            summ, quit_req = run_pour(args, robot, policy, tts, det_cfg, run_dir, run_idx)
            summ["trigger"] = "auto" if args.auto_start else r.source
            announce = True
            session["runs"].append(summ)
            save()
            if quit_req:
                break
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 안전 종료")
        session["interrupted"] = True
    except Exception:
        traceback.print_exc()
        session["error"] = traceback.format_exc()
    finally:
        try:
            if args.backend == "real" and not args.no_return_home and robot.is_connected:
                print("[로봇] 초기 자세로 복귀 (3s) ...")
                robot.return_to_initial()
        except Exception as e:
            print(f"[로봇] 복귀 실패: {e}")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[로봇] disconnect 실패: {e}")
        tts.wait(timeout=6.0)
        tts.close()
        save()
        if args.sim_video and session["runs"]:  # 시뮬 영상에 TTS 소리 입히기
            try:
                sys.path.insert(0, str(GITHUB / "tools"))
                from mux_tts_audio import mux
                mux(Path(args.sim_video), run_dir)
            except Exception as e:
                print(f"[영상] 소리 입히기 실패 (영상은 그대로 있음): {e}")
        print(f"[저장] {run_dir}")
    return 1 if session.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
