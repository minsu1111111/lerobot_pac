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
  IDLE   : TTS ready → 명령 대기. 기본 --stt-mode stream = 최근 2초를 0.5초마다 받아써 바로 반응 ("물 따라줘"),
           (시작은 겹치는 창 2번 연속으로 나와야 함), --stt-mode vad = 말 끝까지 녹음 후 인식,
           --stt-mode enter = Enter 로 녹음 시작/끝. 어느 쪽이든 p/s/q + Enter 키보드 입력도 받음.
           안내 음성이 끝난 뒤에만 듣는다 ("물을 따르겠습니다" 의 '따르' 를 스스로 명령으로 듣지 않게).
  POUR   : pour 명령 → TTS start → policy/processor/감지기 리셋 → 30fps 제어 루프
           감지기(pour_detector)는 '관측된' 오른팔 wrist_roll(state[10]) 을 본다
           DONE → 즉시 TTS done("목표량까지 따랐습니다") → 정책이 정리(내려놓기·복귀)를 마칠 때까지 계속:
             시작 자세 근처(--home-tol-deg 15°)에서 팔·그리퍼가 --settle-s(1s) 동안 멈추면 종료 ("settled"),
             → TTS placed("컵을 놓았습니다"). 최대 --post-done-s(25s)
           --timeout-s(기본 thresholds.json timeout_suggest_s=63) 넘으면 TTS timeout → 정지 (DONE 전일 때만)
  정지 (Space/s, q, 시간 초과, Ctrl+C) — DONE 전이면 '되감기 복귀' (--stop-mode rewind, 기본):
    먼저 물통을 세우고(기울기 15° 미만이던 마지막 자세로 1.5s), 그 자세부터 지나온 경로(보냈던 명령)를
    거꾸로 --rewind-speed(0.7배)로 재생한다.
    → 마지막 동작인 물통 기울이기부터 되돌려 붓기가 먼저 멈추고, 집었던 컵·물통은 집은 자리에 내려놓고
      그리퍼를 연 뒤 시작 자세로 돌아간다. 지나온 길이라 새 경로보다 충돌 위험이 적다.
    되감는 중 Space/s 또는 Ctrl+C → 그 자리에서 즉시 멈춤 (비상).
    DONE 뒤 정지는 되감지 않고 그 자리에서 멈춘다 (되감으면 내려놓은 컵을 다시 집어 온다).
    --stop-mode freeze = 예전 동작 (그 자리에서 멈춤, 모터는 마지막 목표 유지).
  q → 정지 후 프로그램 종료, Ctrl+C → 안전 종료 (real 은 finally 에서 초기자세 복귀 후 disconnect).
  음성 정지: 붓는 동안 별도 프로세스(voice_stop.py)가 듣고 "정지/멈춰/그만" 이면 Space 와 같은 정지 절차.
            제어 루프와 프로세스를 나눠 Whisper 변환이 30fps 를 막지 않는다. --no-voice-stop 으로 끔.

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
from pour_detector import DONE, POURING, READY, PourDetector, PourDetectorConfig  # noqa: E402

FPS = 30
ROLL = JOINT_KEYS.index("right_wrist_roll.pos")  # 10
GRIP = [JOINT_KEYS.index("left_gripper.pos"), JOINT_KEYS.index("right_gripper.pos")]
ARM = [i for i in range(12) if i not in GRIP]
TASK = "Pick up the cup with the left arm and pour water from the bottle into it with the right arm"
# rollout/thresholds.json 이 없을 때의 기본값 = 2026-10 joint_analysis 결과
DEFAULT_TH = dict(
    joint="right_wrist_roll.pos",
    sign=1.0,
    tilt_off=78.88,
    return_off=39.44,
    hold=5,
    base_frames=30,
    timeout_suggest_s=63.0,
)


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
        cmd = {"p": "pour", "": None, "s": "stop", "q": "quit"}.get(line[:1] if line else "")
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
    def __init__(
        self,
        model: str,
        device: str,
        n_action_steps: int | None,
        temporal_ensemble: float | None,
        task: str = TASK,
        keep_backbone_download: bool = False,
        sync_exact: bool = False,
        single: bool = False,
    ):
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
            cfg,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.device = torch.device(device)
        self.task = task
        self.path = path
        # SyncInferenceEngine 은 매 tick 전처리를 돈다. 같은 결과를 더 싸게: ACT 큐 pop 때는 생략
        self.skip_pre_on_pop = not sync_exact and cfg.type == "act" and cfg.temporal_ensemble_coeff is None

        from lerobot.utils.feature_utils import hw_to_dataset_features

        from backends import CAM_SHAPES, MOTORS, _features

        self.single = single
        if single:  # 팔 한 대 정책 (SOFollower: "{motor}.pos" 6 + 카메라 top / wrist). 오른팔 자리에 끼운다.
            self.single_keys = [f"{m}.pos" for m in MOTORS]
            feats = {
                **dict.fromkeys(self.single_keys, float),
                "top": CAM_SHAPES["top"],
                "wrist": CAM_SHAPES["right_wrist"],
            }
            act_keys, self.robot_type = self.single_keys, "so101_follower"
        else:
            # lerobot-rollout 과 같은 방식으로 robot 특징 → dataset 특징 (BiSOFollower 와 같은 키·순서)
            feats, act_keys, self.robot_type = _features(), JOINT_KEYS, "bi_so_follower"
        self.features = {
            **hw_to_dataset_features(dict.fromkeys(act_keys, float), "action"),
            **hw_to_dataset_features(feats, "observation"),
        }
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

        amp = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and self.cfg.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            if self.skip_pre_on_pop and not self.will_infer():
                # ACT select_action 은 큐가 차 있으면 batch 를 보지 않고 popleft 만 한다 (modeling_act.py).
                # 그래서 이미지 전처리(3캠 float 변환·정규화, CPU 10ms+)를 건너뛴다. 결과는 동일.
                a = self.policy.select_action(None)
            else:
                o = obs
                if self.single:  # 12관절 관측 → 오른팔 6관절 + top / 오른손목 카메라
                    o = {k: obs[f"right_{k}"] for k in self.single_keys}
                    o.update(top=obs["top"], wrist=obs["right_wrist"])
                frame = build_dataset_frame(self.features, o, prefix="observation")
                x = prepare_observation_for_inference(frame, self.device, self.task, self.robot_type)
                x = self.pre(x)
                a = self.policy.select_action(x)
            a = self.post(a)
        ad = make_robot_action(a.squeeze(0).cpu(), self.features)
        if self.single:  # 왼팔 칸은 관측값 그대로 (보내지 않음), 오른팔 칸 = 정책 출력
            left = [obs[k] for k in JOINT_KEYS[:6]] if obs is not None else [0.0] * 6
            return np.array(left + [ad[k] for k in self.single_keys], dtype=np.float32)
        return np.array([ad[k] for k in JOINT_KEYS], dtype=np.float32)

    def warmup(self, n=2):
        """첫 추론(CUDA 초기화 등)이 붓기 첫 프레임을 막지 않도록 미리 한 번 돌린다."""
        from backends import CAM_SHAPES

        obs = dict.fromkeys(JOINT_KEYS, 0.0)
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
def match_pause(text: str) -> str | None:
    """일시정지 중 받아쓴 문장 → back / pour / resume / stop (고르는 말이 정지어보다 우선)."""
    sys.path.insert(0, str(GITHUB / "stt"))
    import voice_command as vc

    return vc.match_command(text, priority=vc.PAUSE_PRIORITY)


def clamp_step(cmd, ref, max_deg, max_grip):
    """직전 명령(ref) 대비 프레임당 변화량 제한. 몸통 관절은 도, 그리퍼는 0~100 단위."""
    lim = np.full(12, np.inf, np.float32)
    if max_deg:
        lim[ARM] = max_deg
    if max_grip:
        lim[GRIP] = max_grip
    return ref + np.clip(cmd - ref, -lim, lim)


REWIND_ON = ("manual", "quit", "timeout", "ctrl_c", "voice")  # voice = 붓는 중 "정지/멈춰/그만"


def rewind_plan(
    path,
    sent_path,
    det,
    untilt_deg: float,
    untilt_s: float,
    speed: float,
    home_tol: float | None = None,
    grip_tol: float = 10.0,
) -> tuple[np.ndarray, int]:
    """되감기 목표 궤적 (프레임마다 12관절) 과 '물통 세우기' 단계 길이(프레임).

    목표는 실제로 보냈던 명령(sent_path)을 거꾸로 쓴다. 관측값(path)을 명령하면 실제 로봇에서는
    쥔 그리퍼가 '막힌 위치'를 목표로 받아 쥐는 힘이 빠지고(미끄러짐), 무게로 처진 팔은 더 처진다.
    기울기 판단만 관측값으로 한다.

    1) 물통 세우기: 지금 오른손목이 기준선 대비 untilt_deg 이상 기울어 있으면, 지나온 경로에서 기울기가
       untilt_deg 미만이던 마지막 자세로 untilt_s 동안 바로 보간한다. 붓는 동안 기울인 채 버틴 구간을
       거꾸로 재생하지 않으므로 붓기가 곧바로 멈춘다 (순수 되감기는 시뮬에서 7~20s 걸림).
    2) 되감기: 그 자세부터 경로를 거꾸로 speed 배속으로 (프레임 사이 선형 보간).
    3) 마무리: 붓기 시작 때 관측한 자세로 0.5s.
    home_tol 을 주면 2) 는 마지막으로 시작 자세 근처(팔 관절 모두 home_tol 이내, 그리퍼 grip_tol 이내 = 빈 손)였던
    프레임까지만 되감는다.
    정책이 한 바퀴 돌아 시작 자세로 왔다가 다시 움직인 경우, 그 앞의 경로까지 거꾸로 따라가지 않는다.
    """
    P = np.asarray(sent_path, dtype=np.float32)
    cut = len(P) - 1  # 되감기 시작 프레임
    untilt = []
    if det.baseline is not None:
        tilt = det.cfg.sign * (np.asarray(path, dtype=np.float32)[: len(P), ROLL] - det.baseline)
        tilt_cmd = det.cfg.sign * (P[:, ROLL] - det.baseline)
        if tilt[-1] >= untilt_deg:
            # 관측·명령 둘 다 덜 기울었던 프레임 (팔이 명령을 늦게 따라와 관측만 보면 목표가 아직 기울어 있다)
            low = np.where((tilt < untilt_deg) & (tilt_cmd < untilt_deg))[0]
            if len(low):
                cut = int(low[-1])
                m = max(int(untilt_s * FPS), 1)
                untilt = [P[-1] + (P[cut] - P[-1]) * (i / m) for i in range(1, m + 1)]
    lo = 0
    if home_tol is not None:
        obs = np.asarray(path, dtype=np.float32)[: cut + 1]
        near = np.where(
            (np.abs(obs[:, ARM] - obs[0, ARM]).max(1) < home_tol)
            & (np.abs(obs[:, GRIP] - obs[0, GRIP]).max(1) < grip_tol)
        )[0]
        lo = int(near[-1]) if len(near) else 0
    rev = P[lo : cut + 1][::-1]
    if det.baseline is not None:  # 되감는 중 다시 기울이지 않게: 기울어 있던(붓던) 프레임은 건너뛴다
        T = det.cfg.sign * (np.asarray(path, dtype=np.float32)[: len(P), ROLL] - det.baseline)
        Tc = det.cfg.sign * (P[:, ROLL] - det.baseline)
        keep = ((T < untilt_deg) & (Tc < untilt_deg))[lo : cut + 1][::-1]
        keep[[0, -1]] = True
        rev = rev[keep]
    n = int((len(rev) - 1) / speed) + 1
    back = []
    for k in range(n):
        f = k * speed
        i = min(int(f), len(rev) - 1)
        j = min(i + 1, len(rev) - 1)
        back.append(rev[i] + (rev[j] - rev[i]) * (f - i))
    # 마무리: 첫 '명령'이 아니라 붓기 시작 때 '관측한' 자세로 0.5s 보간. 첫 명령은 정책 첫 출력이라 시작 자세와
    # 다를 수 있다 (학습 안 된 한 팔 정책 시험에서 최대 24° 차이로 멈춤을 확인).
    start = np.asarray(path[0], dtype=np.float32)
    m = int(0.5 * FPS)
    back += [rev[-1] + (start - rev[-1]) * (i / m) for i in range(1, m + 1)]
    return np.asarray(untilt + back, dtype=np.float32), len(untilt)


def rewind(args, robot, path, sent_path, det, keys, rows):
    """정지 후 복귀: 물통 세우기 → 지나온 경로 되감기 (rewind_plan). Space/s·Ctrl+C 면 즉시 멈춤.

    마지막 관측(현재 자세)에서 시작하므로 첫 명령이 튀지 않는다. 명령 변화는 붓기 때와 같은 clamp_step 으로 제한.
    되감으면 집었던 컵·물통은 집은 자리에 내려놓이고(그리퍼 열림) 시작 자세로 돌아간다.
    """
    from lerobot.utils.robot_utils import precise_sleep

    speed = max(args.rewind_speed, 1e-3)
    plan, n_untilt = rewind_plan(
        path,
        sent_path,
        det,
        args.untilt_deg,
        args.untilt_s,
        speed,
        home_tol=args.home_tol_deg,
        grip_tol=args.grip_tol,
    )
    prev = np.asarray(sent_path[-1], dtype=np.float32)  # 현재 모터 목표 → 첫 명령이 튀지 않음
    t0, result, k = time.perf_counter(), "home", -1
    print(
        f"\n[되감기] {'물통 세우기 ' + format(n_untilt / FPS, '.1f') + 's → ' if n_untilt else ''}"
        f"경로 되감기 {(len(plan) - n_untilt) / FPS:.1f}s ({speed:g}배속) | Space/s = 즉시 멈춤"
    )
    try:
        for k, target in enumerate(plan):
            loop_t0 = time.perf_counter()
            key = keys.poll() or ""
            if " " in key or "s" in key or "S" in key:
                result = "aborted"
                break
            sent = clamp_step(target, prev, args.max_step_deg, args.max_step_gripper)
            robot.send_action({kk: float(v) for kk, v in zip(JOINT_KEYS, sent)})
            prev = sent
            phase = "untilt" if k < n_untilt else "rewind"
            if hasattr(robot, "set_overlay"):
                label = "물통 세우기" if phase == "untilt" else "되감기 복귀"
                robot.set_overlay(f"{label} {k / FPS:4.1f}s / {len(plan) / FPS:.1f}s")
            row = dict(step=-1, t=(time.perf_counter() - t0), event=phase)
            row.update({f"cmd_{ii}": float(v) for ii, v in enumerate(sent)})
            rows.append(row)
            if not args.fast:
                dt = time.perf_counter() - loop_t0
                if dt < 1 / FPS:
                    precise_sleep(1 / FPS - dt)
    except KeyboardInterrupt:
        result = "aborted"
    if result == "aborted":
        print("\n[되감기] 비상 멈춤 — 그 자리에서 정지")
    dur = (k + 1) / FPS if args.fast else time.perf_counter() - t0
    return dict(
        result=result,
        steps=k + 1,
        untilt_s=n_untilt / FPS,
        duration_s=dur,
        path_s=len(path) / FPS,
        end_dev_deg=float(
            np.abs(prev[ARM] - np.asarray(path[0])[ARM]).max()
        ),  # 마지막 명령 vs 시작 관측 자세
    )


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return dict(
        n=int(len(x)),
        mean=float(x.mean()),
        p50=float(np.median(x)),
        p95=float(np.percentile(x, 95)),
        p99=float(np.percentile(x, 99)),
        max=float(x.max()),
    )


class DemoPolicy:
    """--policy demo: 모델 대신 재생 중인 데이터셋 에피소드의 action 을 그대로 낸다 (open-loop, replay 전용).
    체크포인트가 오기 전에 감지·TTS·정착·정지/되감기 흐름을 새 작업 데이터로 미리 시험하는 용도."""

    path = "demo"

    def __init__(self):
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(type="demo", chunk_size=1, n_action_steps=1, temporal_ensemble_coeff=None)
        self.robot, self.k = None, 0

    def reset(self):
        self.k = 0

    def will_infer(self) -> bool:
        return False

    def warmup(self, n=2):
        return [0.0] * n

    def __call__(self, obs: dict) -> np.ndarray:
        a = self.robot.demo_action
        out = a[min(self.k, len(a) - 1)].copy()
        self.k += 1
        return out


def dataset_is_single(dataset: Path) -> bool:
    """한 팔(so101_follower) 데이터셋인지: observation.state 이름이 "{motor}.pos" 6개."""
    from backends import MOTORS

    info = json.loads((Path(dataset) / "meta" / "info.json").read_text())
    return info["features"]["observation.state"]["names"] == [f"{m}.pos" for m in MOTORS]


def run_pour(args, robot, policy, tts, det_cfg, run_dir: Path, run_idx: int, voice_stop=None):
    """제어 루프 1회. (요약 dict, quit 여부) 반환. 행 로그는 메모리에 모았다가 끝나고 CSV 로 쓴다."""
    from lerobot.utils.robot_utils import precise_sleep

    period = 1.0 / FPS
    policy.reset()
    det = PourDetector(det_cfg)  # 관측 roll → 실제 판단
    det_cmd = PourDetector(det_cfg)  # 명령 roll → 참고용 (replay 에서 모델이 붓는 동작을 냈는지)
    if hasattr(robot, "restart"):
        robot.restart()

    rows, events_all = [], []
    path: list[np.ndarray] = []  # 실제로 지나온 관절 경로 (관측 state) → 되감기 때 기울기 판단
    sent_path: list[np.ndarray] = []  # 실제로 보낸 명령 → 되감기 목표
    prev_sent = None
    stop_reason, quit_req = None, False
    done_t = None
    still_n = 0  # 시작 자세 근처에서 팔·그리퍼가 거의 안 움직인 연속 프레임 수
    home_n = 0  # 위 + 그리퍼도 시작 때 값 근처 (붓기 없이 복귀 판단용)
    left_home = False  # 팔이 시작 자세에서 크게 벗어난 적이 있는지 (붓기 없이 한 바퀴 돌고 돌아왔는지 판단)
    roll_prev = [None, 0.0]  # [직전 원래 roll, 누적 보정(±360)]
    last_status = -1.0
    warn_n, last_warn = 0, -10.0
    t_start = time.perf_counter()
    step = 0
    test_stop = {"done": False}

    def ev(name, t):
        events_all.append((round(t, 3), name))
        return name

    subtitle = {"text": "", "t": -1e9}  # 시뮬 영상 자막용

    def say(key, t):
        print()  # 상태줄 다음 줄에
        subtitle.update(text=tts.say(key) or key, t=t)
        return ev(f"tts:{key}", t)

    def home_refs() -> list[np.ndarray]:
        refs = [path[0]]
        sp = getattr(args, "start_pose", None)
        if sp is not None and len(sp) in (6, 12):
            r = np.array(path[0], dtype=np.float32)  # 한 팔이면 왼팔 칸은 그대로
            r[12 - len(sp) :] = sp
            refs.append(r)
        return refs

    def read_state():
        obs = robot.get_observation()
        state = np.array([obs[k] for k in JOINT_KEYS], dtype=np.float32)
        # wrist_roll 은 ±180° 에서 부호가 뒤집힌다 → 이어 붙인 연속 각도로 감지기·되감기에 쓴다.
        # (기준 자세가 큰 팔에서 붓다가 180° 를 넘으면 '복귀' 로 오판해 중간에 DONE 이 났다: follower1 실측)
        raw_roll = float(state[ROLL])
        if roll_prev[0] is not None and abs(raw_roll - roll_prev[0]) > 180:
            roll_prev[1] -= 360 if raw_roll > roll_prev[0] else -360
        roll_prev[0] = raw_roll
        state[ROLL] = raw_roll + roll_prev[1]
        return obs, state

    def pause(t, keys) -> str:
        """붓는 중 정지 (--stop-mode ask): 기울어 있으면 세운 뒤 멈춰서 기다린다.
        "돌아가"(Space·r) → back = 되감기, "계속 해줘"(c) → resume = 정책 이어서, q → quit,
        --pause-s 동안 아무 말 없으면 timeout (= 되감기). 세우는 동안의 관측·명령도 path·sent_path 에 넣어
        나중에 되감기가 지금 자세에서 시작하게 한다 (rewind_plan 은 기울어 있던 프레임을 건너뛴다)."""
        nonlocal prev_sent, t_start
        t0 = time.perf_counter()
        if voice_stop is not None:  # 안내 음성이 명령으로 들리지 않게 잠시 끔
            voice_stop.disarm()
        # 안내는 세우기가 끝난 뒤에: "멈춰" 직후 바로 말하면 어색하다 (사용자 의견). 멈추고 세우는 동작이
        # 먼저 반응이 되고, 최소 --pause-say-s 뒤에 말한다.
        plan, n_untilt = rewind_plan(path, sent_path, det, args.untilt_deg, args.untilt_s, 1.0)
        # 세우기 뒤 최대 1s: 팔이 마지막 목표를 따라올 때까지 같은 목표 유지 (관측이 아직 기울어 있으면
        # 되감기가 세우기를 한 번 더 한다)
        hold = [plan[n_untilt - 1]] * FPS if n_untilt else []
        for i, target in enumerate(list(plan[:n_untilt]) + hold):
            if (
                i >= n_untilt
                and det.baseline is not None
                and det.cfg.sign * (path[-1][ROLL] - det.baseline) < args.untilt_deg
            ):
                break
            lt = time.perf_counter()
            _, state = read_state()
            sent = clamp_step(target, prev_sent, args.max_step_deg, args.max_step_gripper)
            out = robot.send_action({k: float(v) for k, v in zip(JOINT_KEYS, sent)})
            if isinstance(out, dict) and all(k in out for k in JOINT_KEYS):
                sent = np.array([out[k] for k in JOINT_KEYS], dtype=np.float32)
            path.append(state)
            sent_path.append(sent)
            prev_sent = sent
            row = dict(step=-1, t=t, event="pause_untilt")
            row.update({f"obs_{i}": float(v) for i, v in enumerate(state)})
            row.update({f"cmd_{i}": float(v) for i, v in enumerate(sent)})
            rows.append(row)
            if hasattr(robot, "set_overlay"):
                robot.set_overlay("정지 · 물통 세우기")
            if not args.fast:
                dt = time.perf_counter() - lt
                if dt < 1 / FPS:
                    precise_sleep(1 / FPS - dt)
        if not args.fast:
            time.sleep(max(0.0, args.pause_say_s - (time.perf_counter() - t0)))
        say("stopped", t)
        tts.wait(timeout=8.0)
        time.sleep(0.3)  # 스피커 잔향
        if voice_stop is not None:
            voice_stop.arm()
        print(
            "[일시정지] '돌아가' → 되감기 / '계속 해줘' → 이어서  (키: Space·r = 돌아가, c = 계속, q = 종료)"
        )
        if hasattr(robot, "set_overlay"):
            robot.set_overlay("정지 · '돌아가' / '계속 해줘' 기다리는 중")
        choice, tw = None, time.perf_counter()
        while choice is None:
            k = keys.poll() or ""
            heard = voice_stop.poll_command() if voice_stop is not None else None
            if heard:  # 일시정지 우선순위로 다시 판정 ("그만 돌아가" → back)
                heard = (match_pause(heard[1]) or heard[0], heard[1])
            waited = time.perf_counter() - tw
            if "q" in k or "Q" in k:
                choice = "quit"
            elif any(c in k for c in " rRsS") or (heard and heard[0] == "back"):
                choice = "back"
            elif "c" in k or "C" in k or (heard and heard[0] in ("resume", "pour")):
                choice = "resume"
            elif args.test_pause_cmd and (args.fast or waited >= 1.0):
                choice = args.test_pause_cmd
            elif waited >= args.pause_s:
                choice = "timeout"
            else:
                time.sleep(0.03)
        print(f"[일시정지] → {choice} ({time.perf_counter() - t0:.1f}s 멈춤)")
        ev(f"pause:{choice}", t)
        if choice == "resume":
            policy.reset()  # 멈추기 전 행동 묶음을 버리고 지금 장면에서 새로 계산
            if det.state == POURING:  # 세워서 기울기가 돌아온 걸 '다 따름'으로 보지 않게
                det.state, det._cnt = READY, 0
            if not args.fast:  # 시간 초과(--timeout-s)는 멈춰 있던 시간을 빼고 센다
                t_start += time.perf_counter() - t0
        return choice

    with RawKeys() as keys:
        if not keys.enabled:
            print("[키] stdin 이 터미널이 아님 → Space/s/q 정지 비활성 (Ctrl+C 만 가능)")

        def _loop():
            nonlocal \
                stop_reason, \
                quit_req, \
                done_t, \
                last_status, \
                warn_n, \
                last_warn, \
                step, \
                prev_sent, \
                still_n, \
                home_n, \
                left_home
            evs = [say("start", 0.0)]
            while True:
                loop_t0 = time.perf_counter()
                t = step / FPS if args.fast else loop_t0 - t_start

                # ---- 수동 정지 (키) ----
                k = keys.poll() or ""
                if args.test_stop_at is not None and t >= args.test_stop_at and not test_stop["done"]:
                    k += "s"  # 시험용: 정지 키를 누른 것처럼 (한 번만)
                    test_stop["done"] = True
                if "q" in k or "Q" in k:
                    stop_reason, quit_req = "quit", True
                elif " " in k or "s" in k or "S" in k:
                    stop_reason = "manual"
                elif voice_stop is not None and voice_stop.poll():
                    stop_reason = "voice"
                if (
                    stop_reason
                    and args.stop_mode == "ask"
                    and stop_reason in ("manual", "voice")
                    and done_t is None
                ):
                    evs.append(ev(f"stop:{stop_reason}", t))
                    rows.append(dict(step=step, t=t, event=";".join(evs)))
                    evs = []
                    choice = pause(t, keys)
                    if choice == "resume":
                        stop_reason = None
                        evs.append(say("resume", t))
                        continue
                    if choice == "quit":
                        stop_reason, quit_req = "quit", True
                    break
                if stop_reason:
                    evs.append(say("stopped", t))
                    evs.append(ev(f"stop:{stop_reason}", t))
                    rows.append(dict(step=step, t=t, event=";".join(evs)))
                    break

                # ---- 관측 → 정책 → 행동 ----
                obs, state = read_state()
                t_obs = time.perf_counter()
                path.append(state)
                new_chunk = policy.will_infer()
                cmd = policy(obs)
                t_inf = time.perf_counter()
                ref = prev_sent if prev_sent is not None else state
                sent = clamp_step(cmd, ref, args.max_step_deg, args.max_step_gripper)
                out = robot.send_action({k: float(v) for k, v in zip(JOINT_KEYS, sent)})
                # 실제로 보낸 값을 되감기 경로로 (lerobot --max-relative-target 이 자르면 반환값이 다르다)
                if isinstance(out, dict) and all(k in out for k in JOINT_KEYS):
                    sent = np.array([out[k] for k in JOINT_KEYS], dtype=np.float32)
                sent_path.append(sent)
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
                        lines.append(f"완료 후 {t - done_t:4.1f}s (정리 중, 최대 {args.post_done_s:.0f}s)")
                    if t - subtitle["t"] < 4.0:
                        lines.append(f"[TTS] {subtitle['text']}")
                    robot.set_overlay("\n".join(lines))

                work = t_act - loop_t0
                row = dict(
                    step=step,
                    t=t,
                    work_ms=work * 1e3,
                    obs_ms=(t_obs - loop_t0) * 1e3,
                    infer_ms=(t_inf - t_obs) * 1e3,
                    new_chunk=int(new_chunk),
                    det=det.state,
                    det_cmd=det_cmd.state,
                    roll_obs=float(state[ROLL]),
                    roll_cmd=float(cmd[ROLL]),
                    roll_sent=float(sent[ROLL]),
                    jump_arm=float(jump[ARM].max()),
                    jump_roll=float(jump[ROLL]),
                    jump_grip=float(jump[GRIP].max()),
                    clamped=int(np.any(np.abs(sent - cmd) > 1e-4)),
                )
                row.update({f"obs_{i}": float(v) for i, v in enumerate(state)})
                row.update({f"cmd_{i}": float(v) for i, v in enumerate(sent)})

                # ---- 종료 조건 ----
                # DONE 뒤: 정책이 정리(내려놓기·복귀)를 끝내고 멈췄으면 종료. 시연은 모두 정지 자세로 끝난다.
                # 그리퍼도 본다: 팔은 멈춘 채 그리퍼만 열어 내려놓는 구간이 있다. 시작 자세 근처일 때만 끝으로 본다.
                # DONE 전: 붓지 않고(기울기 부족 등) 한 바퀴 돌아 시작 자세로 돌아와 멈췄으면 거기서 끝낸다.
                # 안 그러면 정책이 처음부터 다시 집으러 가고, 시간초과 되감기가 그 전체를 거꾸로 따라간다
                # (실제 follower1 60k: 26s 에 복귀 → 다시 집기 → 40s 시간초과 → 되감기 40s 분량).
                if len(path) > 1:
                    d = np.abs(path[-1] - path[-2])
                    still = d[ARM].max() < args.settle_deg and d[GRIP].max() < args.settle_deg
                    left_home = left_home or (
                        np.abs(path[-1][ARM] - path[0][ARM]).max() > args.home_tol_deg + args.away_deg
                    )
                    # '시작 자세' = 붓기 시작 때 자세 또는 시연의 쉬는 자세 (기준값 파일 start_pose) 중 가까운 쪽.
                    # 팔을 다른 자세에 둔 채 시작해도 정책은 학습한 쉬는 자세로 돌아간다 (follower1 실측: 34° 차이로
                    # 정착 판정이 안 돼 "컵을 놓았습니다" 없이 25s 뒤 끝남).
                    homes = home_refs()
                    dev = min(np.abs(path[-1][ARM] - h[ARM]).max() for h in homes)
                    # 그리퍼도 시작 때 값 근처여야 (컵을 쥔 채 시작 자세를 지나가는 건 '돌아옴'이 아님)
                    grip_home = min(np.abs(path[-1][GRIP] - h[GRIP]).max() for h in homes) < args.grip_tol
                    still_n = still_n + 1 if (still and dev < args.home_tol_deg) else 0
                    home_n = home_n + 1 if (still and dev < args.home_tol_deg and grip_home) else 0
                if done_t is not None and t - done_t >= 2.0 and still_n >= args.settle_s * FPS:
                    stop_reason = "settled"
                    evs.append(say("placed", t))  # 정리(컵 내려놓기·복귀)까지 끝남
                elif done_t is None and left_home and home_n >= args.settle_s * FPS:
                    if det.tilt_step is not None:
                        # 붓다가 정지 → "계속" 뒤 정책이 다시 기울이지 않고 내려놓고 돌아온 경우: 이미 부었으니 정상 종료
                        # (양팔 시뮬: 거의 다 부은 25s 에 정지 → 계속 → 다시 안 붓고 복귀)
                        stop_reason = "settled"
                        evs.append(say("placed", t))
                    else:
                        stop_reason = "no_pour"  # 이미 시작 자세 → 되감기 안 함
                        evs.append(say("stopped", t))
                elif done_t is not None and t - done_t >= args.post_done_s:
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
                    line = (
                        f"[{t:6.1f}s] det={det.state:7s} roll obs {state[ROLL]:7.1f} cmd {cmd[ROLL]:7.1f}  "
                        f"work {work * 1e3:5.1f}ms infer {(t_inf - t_obs) * 1e3:6.1f}ms  jump {jump[ARM].max():5.1f}"
                    )
                    if done_t is not None:
                        line += f"  완료 후 {t - done_t:4.1f}s"
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
                            print(
                                f"\n[경고] 루프 지연 {dt * 1e3:.0f}ms > {period * 1e3:.0f}ms "
                                f"(추론 {(t_inf - t_obs) * 1e3:.0f}ms, 누적 {warn_n}회)"
                            )
                row["period_ms"] = (time.perf_counter() - loop_t0) * 1e3
                rows.append(row)
                step += 1

        if voice_stop is not None:
            voice_stop.arm()
        try:
            _loop()
        except KeyboardInterrupt:  # 행동 전송을 즉시 멈추고 로그는 저장한 뒤 프로그램 종료
            stop_reason, quit_req = "ctrl_c", True
            tts.say("stopped")
            events_all.append((round(time.perf_counter() - t_start, 3), "stop:ctrl_c"))
        finally:
            if voice_stop is not None:  # "정지했습니다" 안내를 다시 듣지 않게 바로 끈다
                voice_stop.disarm()
        rewind_info = None
        if (
            args.stop_mode in ("rewind", "ask")
            and stop_reason in REWIND_ON
            and done_t is None
            and len(sent_path) > 1
        ):
            t_rw = (step / FPS) if args.fast else time.perf_counter() - t_start
            events_all.append((round(t_rw, 3), "rewind:start"))
            tts.say("going_back")  # "원래 자리로 돌아가겠습니다." (돌아가 / 시간초과 / 종료)
            rewind_info = rewind(args, robot, path, sent_path, det, keys, rows)
            events_all.append((round(t_rw + rewind_info["duration_s"], 3), f"rewind:{rewind_info['result']}"))
            if rewind_info["result"] == "home":
                tts.say("returned")
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
        run=run_idx,
        csv=str(csv_path),
        stop_reason=stop_reason,
        steps=len(R),
        duration_s=(R[-1]["t"] if R else 0.0),
        wall_s=time.perf_counter() - t_start,
        done_t=done_t,
        done_reason=det.done_reason,
        tilt_t=(det.tilt_step / FPS if det.tilt_step is not None else None),
        cmd_detector=dict(
            state=det_cmd.state,
            done_t=(det_cmd.done_step / FPS if det_cmd.done_step is not None else None),
            tilt_t=(det_cmd.tilt_step / FPS if det_cmd.tilt_step is not None else None),
        ),
        baseline=det.baseline,
        period_ms=stats([r["period_ms"] for r in R if "period_ms" in r]),
        work_ms=stats([r["work_ms"] for r in R]),
        infer_ms_chunk=stats([r["infer_ms"] for r in R if r["new_chunk"]]),
        infer_ms_pop=stats([r["infer_ms"] for r in R if not r["new_chunk"]]),
        overruns=int(sum(r["work_ms"] > period * 1e3 for r in R)),
        jump_arm_boundary=stats(jumps_b),
        jump_arm_within=stats(jumps_i),
        jump_arm_all=stats([r["jump_arm"] for r in R[1:]]),
        jump_roll_all=stats([r["jump_roll"] for r in R[1:]]),
        jump_grip_all=stats([r["jump_grip"] for r in R[1:]]),
        clamped_steps=int(sum(r["clamped"] for r in R)),
        rewind=rewind_info,
        end_dev_from_start=(float(np.abs(path[-1][ARM] - path[0][ARM]).max()) if path else None),
        events=events_all,
    )
    if hasattr(robot, "decode_wait_s"):
        summ["replay_decode_wait_s"] = robot.decode_wait_s
    if hasattr(robot, "state") and hasattr(robot, "episode"):  # 시연 자체의 DONE (참고)
        d = PourDetector(det_cfg).run(robot.state[:, ROLL])
        summ["demo_done_t"] = d.done_step / FPS if d.state == DONE else None
        summ["episode"], summ["episode_len_s"] = robot.episode, robot.length / FPS
    if rewind_info:
        print(
            f"[되감기] {rewind_info['result']}  {rewind_info['duration_s']:.1f}s  "
            f"시작 자세와 차이(팔 최대) {rewind_info['end_dev_deg']:.1f}°"
        )
    print(
        f"[결과] 정지={stop_reason}  DONE={done_t if done_t is None else round(done_t, 2)}s  "
        f"steps={len(R)}  지연 {summ['overruns']}회  "
        f"점프(팔, 최대) 경계 {summ['jump_arm_boundary']['max'] if jumps_b else 0:.1f} / 내부 "
        f"{summ['jump_arm_within']['max'] if jumps_i else 0:.1f}  → {csv_path}"
    )
    return summ, quit_req


# --------------------------------------------------------------------------- #
def load_thresholds(path: Path) -> dict:
    if path.is_file():
        return {**DEFAULT_TH, **json.loads(path.read_text())}
    print(f"[감지기] {path} 없음 → 내장 기본값 사용")
    return dict(DEFAULT_TH)


def make_robot(args):
    from backends import DatasetReplayRobot, RealRobot, ReplayMujocoRobot, SingleArmRobot, install_calib

    clock = "step" if args.fast else args.replay_clock
    if args.backend == "replay":
        return DatasetReplayRobot(args.episode, Path(args.dataset), clock=clock)
    if args.backend == "replay+mujoco":
        return ReplayMujocoRobot(
            args.episode,
            Path(args.dataset),
            clock=clock,
            render=args.sim_render,
            video_path=args.sim_video,
            spacing=args.sim_spacing,
            yaw_deg=args.sim_yaw,
        )
    if args.backend == "single":
        if not (args.top_cam and args.wrist_cam):
            raise SystemExit("single 백엔드는 --top-cam 과 --wrist-cam 이 필요")
        return SingleArmRobot(
            args.single_port,
            args.single_id,
            args.top_cam,
            args.wrist_cam,
            fourcc=args.fourcc,
            max_relative_target=args.max_relative_target,
            keep_torque=args.keep_torque,
        )
    if args.calib_left or args.calib_right:
        install_calib(args.robot_id, args.calib_left, args.calib_right)
    need = dict(
        left_port=args.left_port,
        right_port=args.right_port,
        top_cam=args.top_cam,
        left_cam=args.left_cam,
        right_cam=args.right_cam,
    )
    missing = [k for k, v in need.items() if not v]
    if missing:
        raise SystemExit(f"real 백엔드 인자 부족: {missing} (run_demo.sh / robot.env 참고)")
    return RealRobot(
        args.robot_id,
        max_relative_target=args.max_relative_target,
        keep_torque=args.keep_torque,
        fourcc=args.fourcc,
        **need,
    )


def parse_args(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_argument_group("백엔드")
    g.add_argument("--backend", choices=["real", "replay", "replay+mujoco", "single"], default="replay")
    g.add_argument("--episode", type=int, default=0, help="replay 에피소드 번호")
    g.add_argument("--dataset", default=str(DATASET))
    g.add_argument("--replay-clock", choices=["step", "wall"], default="step")
    g.add_argument(
        "--fast", action="store_true", help="잠 안 자고 최대 속도 (시간 = step/30, replay 전용 측정용)"
    )
    g.add_argument("--sim-render", choices=["none", "window", "offscreen"], default="none")
    g.add_argument(
        "--sim-spacing",
        type=float,
        default=0.45,
        help="시뮬 두 팔 받침 사이 거리(m). 기본 0.45 = 9/20 데이터 추정. 10/9 리그 탑 카메라 기준 추정 ≈0.32",
    )
    g.add_argument(
        "--sim-yaw", type=float, default=30.0, help="시뮬 두 팔을 안쪽으로 돌린 각도(°). 10/9 리그 추정 ≈0"
    )
    g.add_argument("--sim-video", default=None, help="replay+mujoco 영상 저장 경로 (offscreen)")
    g = ap.add_argument_group("real 로봇")
    g.add_argument(
        "--robot-id",
        default=os.environ.get("ROBOT_ID", "bimanual"),
        help="BiSOFollower id → 캘리브레이션 {id}_left.json / {id}_right.json",
    )
    g.add_argument("--left-port", default=os.environ.get("LEFT_PORT"))
    g.add_argument("--right-port", default=os.environ.get("RIGHT_PORT"))
    g.add_argument("--top-cam", default=os.environ.get("TOP_CAM"))
    g.add_argument("--left-cam", default=os.environ.get("LEFT_CAM"))
    g.add_argument("--right-cam", default=os.environ.get("RIGHT_CAM"))
    g.add_argument("--fourcc", default=os.environ.get("CAM_FOURCC") or None)
    g = ap.add_argument_group("single 로봇 (팔 한 대 리허설: 그 팔을 오른팔=붓는 팔 자리에 둠)")
    g.add_argument("--single-port", default="/dev/follower1")
    g.add_argument("--single-id", default="follower1", help="캘리브레이션 so_follower/<id>.json")
    g.add_argument("--wrist-cam", default=None, help="그 팔의 손목 카메라 (top 은 --top-cam)")
    g.add_argument("--calib-left", default=os.environ.get("CALIB_LEFT"), help="예: follower1 (복사 원본)")
    g.add_argument("--calib-right", default=os.environ.get("CALIB_RIGHT"), help="예: follower2")
    g.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="lerobot 안전 제한: 현재 위치 대비 목표 최대 차이 (도). 기본 없음(학습 때와 같음)",
    )
    g.add_argument(
        "--keep-torque", action="store_true", help="종료 때 토크 유지 (기본: lerobot 처럼 토크 끔)"
    )
    g.add_argument("--no-return-home", action="store_true", help="종료 때 초기 자세 복귀 생략")
    g = ap.add_argument_group("정책")
    g.add_argument(
        "--policy",
        default=MODEL_REPO,
        help="HF repo id 또는 로컬 폴더. demo = 데이터셋 action 재생 (replay 전용, 모델 없이 흐름 시험)",
    )
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--n-action-steps", type=int, default=None, help="청크에서 실제 쓰는 개수 (학습값 100)")
    g.add_argument(
        "--temporal-ensemble",
        type=float,
        default=None,
        metavar="COEF",
        help="ACT temporal ensembling (원 논문 0.01). n_action_steps=1, 매 프레임 추론",
    )
    g.add_argument(
        "--max-step-deg",
        type=float,
        default=10.0,
        help="몸통 관절 프레임당 명령 변화 제한 (도). 기본 10, 0 이면 끔",
    )
    g.add_argument(
        "--max-step-gripper", type=float, default=None, help="그리퍼 프레임당 명령 변화 제한 (0~100)"
    )
    g.add_argument("--task", default=TASK)
    g.add_argument("--threads", type=int, default=None, help="torch CPU 스레드 수 (기본: torch 기본값)")
    g.add_argument(
        "--sync-exact",
        action="store_true",
        help="SyncInferenceEngine 처럼 매 프레임 전처리 (기본: 큐 pop 프레임은 생략, 결과 동일)",
    )
    g = ap.add_argument_group("데모 흐름")
    g.add_argument("--thresholds", default=str(THRESHOLDS))
    g.add_argument(
        "--post-done-s",
        type=float,
        default=25.0,
        help="DONE 뒤 정책을 더 돌리는 최대 시간 (보통 settled 로 먼저 끝남)",
    )
    g.add_argument("--settle-s", type=float, default=1.0, help="DONE 뒤 팔이 이 시간 동안 멈춰 있으면 종료")
    g.add_argument(
        "--settle-deg", type=float, default=0.5, help="'멈춤' 판정: 프레임당 관절 변화(도·그리퍼) 상한"
    )
    g.add_argument(
        "--away-deg",
        type=float,
        default=30.0,
        help="DONE 전 팔이 home-tol+이 값 넘게 벗어났다가 빈 손으로 시작 자세에 돌아와 멈추면 종료 (no_pour, 되감기 없음)",
    )
    g.add_argument(
        "--grip-tol",
        type=float,
        default=10.0,
        help="'빈 손 = 시작 때 그리퍼 값' 판정 폭 (no_pour 종료, 되감기 시작점)",
    )
    g.add_argument(
        "--home-tol-deg",
        type=float,
        default=15.0,
        help="'멈춤' 은 시작 자세와 팔 관절 차이가 이 이내일 때만 인정",
    )
    g.add_argument(
        "--stop-mode",
        choices=["ask", "rewind", "freeze"],
        default="ask",
        help="DONE 전 정지 때: ask = 세운 뒤 멈춰서 '돌아가'(되감기)/'계속 해줘'(이어서) 기다림, "
        "rewind = 바로 되감기, freeze = 그 자리에서 멈춤",
    )
    g.add_argument("--pause-s", type=float, default=20.0, help="ask: 이 시간 동안 아무 말 없으면 되감기 복귀")
    g.add_argument(
        "--pause-say-s",
        type=float,
        default=0.8,
        help="ask: 정지 뒤 '정지했습니다.' 를 말하기까지 최소 시간 (물통 세우기가 더 길면 세운 뒤)",
    )
    g.add_argument(
        "--test-pause-cmd",
        choices=["back", "resume"],
        default=None,
        help="시험용: 정지 뒤 이 명령을 들은 것처럼",
    )
    g.add_argument("--rewind-speed", type=float, default=0.7, help="되감기 배속 (1 = 원래 속도)")
    g.add_argument(
        "--untilt-deg",
        type=float,
        default=15.0,
        help="정지 때 오른손목이 기준선 대비 이 이상 기울어 있으면 먼저 세움",
    )
    g.add_argument("--untilt-s", type=float, default=1.5, help="물통 세우기에 쓰는 시간 (초)")
    g.add_argument("--timeout-s", type=float, default=None, help="기본: thresholds.json timeout_suggest_s")
    g.add_argument("--no-stt", action="store_true")
    g.add_argument("--no-tts", action="store_true")
    g.add_argument("--stt-model", default="small")
    g.add_argument(
        "--stt-mode",
        choices=("stream", "vad", "enter"),
        default="stream",
        help="stream = 최근 2초를 0.5초마다 받아써 바로 반응 (기본, 소음에 강함), "
        "vad = 말 끝까지 녹음 후 인식 (소음이 계속되면 최대 녹음 길이까지 기다림), enter = Enter 로 녹음 시작/끝",
    )
    g.add_argument(
        "--vad-level",
        type=int,
        default=2,
        choices=range(4),
        help="VAD 민감도 0(관대)~3(엄격). 시끄러우면 높임",
    )
    g.add_argument("--silence-ms", type=int, default=1000, help="vad: 이 시간 무음이면 말 끝으로 봄")
    g.add_argument("--mic", default=None)
    g.add_argument("--no-voice-stop", action="store_true", help="붓는 동안 음성 정지 끔 (키보드 정지만)")
    g.add_argument(
        "--voice-stop-hop-s",
        type=float,
        default=0.5,
        help="음성 정지: 최근 2s 를 이 간격마다 확인 (짧을수록 빨리 반응, GPU 더 씀)",
    )
    g.add_argument("--auto-start", action="store_true", help="명령 없이 바로 1회 붓고 종료 (시험용)")
    g.add_argument("--test-stop-at", type=float, default=None, help="시험용: 이 시각(s)에 정지 키 입력 흉내")
    g.add_argument("--out", default=str(OUTPUTS / "rollout"))
    g.add_argument("--tag", default="")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    th = load_thresholds(Path(args.thresholds))
    # 한 팔: 실제 팔(--backend single) 또는 한 팔 데이터셋을 재생하는 시뮬(replay / replay+mujoco)
    single = args.backend == "single" or (args.backend != "real" and dataset_is_single(Path(args.dataset)))
    if args.policy == "demo" and args.backend not in ("replay", "replay+mujoco"):
        raise SystemExit("--policy demo 는 replay / replay+mujoco 전용 (데이터셋 action 재생)")
    # 감지기는 state[10] = 오른손목 roll 을 본다. 한 팔은 그 팔을 오른팔 자리에 넣으므로 "wrist_roll.pos".
    assert th["joint"] == ("wrist_roll.pos" if single else "right_wrist_roll.pos"), th["joint"]
    if args.timeout_s is None:
        args.timeout_s = float(th["timeout_suggest_s"])
    args.start_pose = th.get("start_pose")  # 시연의 쉬는 자세 (정착 판정에 함께 씀, 없으면 붓기 시작 자세만)
    det_cfg = PourDetectorConfig(
        tilt_off=th["tilt_off"],
        return_off=th["return_off"],
        hold=th["hold"],
        base_frames=th["base_frames"],
        sign=th["sign"],
    )
    if args.fast and args.backend == "real":
        raise SystemExit("--fast 는 replay 전용")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / (stamp + (f"_{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    print(f"[정책] {args.policy} 로딩 (device={args.device}) ...")
    policy = (
        DemoPolicy()
        if args.policy == "demo"
        else Policy(
            args.policy,
            args.device,
            args.n_action_steps,
            args.temporal_ensemble,
            args.task,
            sync_exact=args.sync_exact,
            single=single,
        )
    )
    cfg = policy.cfg
    print(
        f"[정책] chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
        f"temporal_ensemble={cfg.temporal_ensemble_coeff} ← {policy.path}"
    )
    if cfg.n_action_steps >= 100 and cfg.temporal_ensemble_coeff is None and not args.max_step_deg:
        print(
            "[경고] n_action_steps=100 (학습값): 청크 경계마다 명령이 한 프레임에 크게 튈 수 있음 "
            "(replay open-loop 측정: 팔 관절 최대 18~70°/프레임, 시연은 ≤10°). --n-action-steps / --temporal-ensemble / --max-step-deg 참고"
        )
    wt = policy.warmup()
    print(f"[정책] 워밍업 추론 {', '.join(f'{x * 1e3:.0f}ms' for x in wt)}")
    if cfg.temporal_ensemble_coeff is not None and wt[-1] > 1 / FPS and not args.fast:
        print(f"[경고] temporal ensemble 은 매 프레임 추론 ({wt[-1] * 1e3:.0f}ms > 33ms) → 30fps 불가")

    robot = make_robot(args)
    if isinstance(policy, DemoPolicy):
        policy.robot = robot
    tts = make_tts(not args.no_tts)
    commander = None if args.auto_start else make_commander(not args.no_stt, args.stt_model, args.mic)
    voice_stop = None
    if not args.no_stt and not args.no_voice_stop:
        from voice_stop import VoiceStop

        print("[음성정지] 붓는 동안 듣는 프로세스 시작 (Whisper 로딩) ...")
        voice_stop = VoiceStop(model=args.stt_model, mic=args.mic, hop_s=args.voice_stop_hop_s)
        if not voice_stop.start():
            voice_stop.close()
            voice_stop = None
    session = dict(
        args=vars(args),
        thresholds=th,
        started=stamp,
        device=args.device,
        policy_path=policy.path,
        n_action_steps=cfg.n_action_steps,
        temporal_ensemble=cfg.temporal_ensemble_coeff,
        warmup_ms=[x * 1e3 for x in wt],
        runs=[],
        idle_events=[],
    )

    def save():
        (run_dir / "summary.json").write_text(json.dumps(session, indent=2, ensure_ascii=False, default=str))

    try:
        print(f"[로봇] {args.backend} 연결 ...")
        robot.connect()
        run_idx, announce = 0, True
        if args.stt_mode == "stream" and hasattr(commander, "listen_stream"):

            def listen():
                return commander.listen_stream()
        elif args.stt_mode == "vad" and hasattr(commander, "listen_auto"):

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
                session["idle_events"].append(
                    dict(t=time.time(), text=r.text, command=r.command, source=r.source, stt_s=r.stt_s)
                )
                if r.command == "quit":
                    break
                if r.command != "pour":
                    # 자동 감지는 소음에도 켜질 수 있어, 받아쓴 말이 있을 때만 다시 말해 달라고 한다
                    if r.command is None and (args.stt_mode == "enter" or (r.text or "").strip()):
                        tts.say("retry")
                    continue
            # ---- POUR ----
            run_idx += 1
            summ, quit_req = run_pour(args, robot, policy, tts, det_cfg, run_dir, run_idx, voice_stop)
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
            if args.backend in ("real", "single") and not args.no_return_home and robot.is_connected:
                print("[로봇] 초기 자세로 복귀 (3s) ...")
                robot.return_to_initial()
        except Exception as e:
            print(f"[로봇] 복귀 실패: {e}")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[로봇] disconnect 실패: {e}")
        if voice_stop is not None:
            voice_stop.close()
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
