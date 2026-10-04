"""TTS 가 30fps 제어 루프를 막지 않는지 확인하는 가짜 루프 (로봇 없이, 실시간 속도).

데이터셋 observation.state 를 30fps 로 재생하며 PourDetector 를 돌리고,
에피소드 시작에 say("start"), DONE 에 say("done") 을 부른다. 영상 디코딩 없음, 원본 수정 없음.
주기 측정: 매 반복 시작 시각(perf_counter) 차이. 대기는 lerobot precise_sleep 으로 마감 시각까지.
--work-ms 를 주면 매 프레임 그만큼 파이썬 연산(GIL 점유)을 넣어 정책 추론 부하를 흉내 낸다.

tts/ 폴더(독립 모듈)와 이 프로젝트의 데이터셋·감지기를 묶는 통합 시험이라 tools/ 에 둔다.

실행:
  python tools/tts_loop_timing.py                              # 에피소드 0,1
  python tools/tts_loop_timing.py --episodes 0 1 2 --max-seconds 60 --work-ms 15
결과: local/outputs/tts/loop_timing.txt
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASET, GITHUB, OUTPUTS, THRESHOLDS  # noqa: E402
from pour_detector import DONE, PourDetector, PourDetectorConfig  # noqa: E402

sys.path.insert(0, str(GITHUB / "tts"))
from tts_player import TTSPlayer  # noqa: E402

try:
    from lerobot.utils.robot_utils import precise_sleep
except ImportError:  # lerobot 없이도 돌도록 (Linux 에서는 time.sleep 과 동일)
    def precise_sleep(seconds: float):
        if seconds > 0:
            time.sleep(seconds)

FPS = 30
JOINT_IDX = 10  # right_wrist_roll.pos (12차원 state 내 위치)


def load_states(episodes: list[int]) -> dict[int, np.ndarray]:
    files = sorted(glob.glob(str(DATASET / "data" / "chunk-*" / "file-*.parquet")))
    cols = ["episode_index", "frame_index", "observation.state"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
    df = df[df.episode_index.isin(episodes)].sort_values(["episode_index", "frame_index"])
    return {e: np.stack(g["observation.state"].to_numpy()) for e, g in df.groupby("episode_index")}


def detector_cfg() -> PourDetectorConfig:
    s = json.loads(THRESHOLDS.read_text())
    assert s["joint"] == "right_wrist_roll.pos"
    return PourDetectorConfig(tilt_off=s["tilt_off"], return_off=s["return_off"], hold=s["hold"],
                              base_frames=s["base_frames"], sign=s["sign"],
                              timeout_steps=int(s.get("timeout_suggest_s", 63) * FPS))


def busy(ms: float):
    end = time.perf_counter() + ms / 1000
    x = 0
    while time.perf_counter() < end:
        x += 1


def stats(p_ms: np.ndarray, period_ms: float) -> str:
    if len(p_ms) == 0:
        return "n=0"
    over = int((p_ms > 1.5 * period_ms).sum())
    return (f"n={len(p_ms):5d}  mean={p_ms.mean():6.2f}  std={p_ms.std():5.2f}  "
            f"p99={np.percentile(p_ms, 99):6.2f}  max={p_ms.max():6.2f} ms  overrun(>{1.5 * period_ms:.1f}ms)={over}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-seconds", type=float, default=None, help="에피소드당 최대 재생 시간")
    ap.add_argument("--work-ms", type=float, default=0.0, help="프레임당 가짜 연산 시간")
    ap.add_argument("--no-audio", action="store_true", help="자막만 (비교용)")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "tts" / "loop_timing.txt")
    args = ap.parse_args()

    states = load_states(args.episodes)
    cfg = detector_cfg()
    dt = 1.0 / FPS
    tts = TTSPlayer(enabled=not args.no_audio)

    starts, says, say_cost, events = [], [], [], []  # 반복 시작 시각 / say 호출 시각 / say 소요
    t_next = time.perf_counter()
    for ep in args.episodes:
        x = states[ep][:, JOINT_IDX]
        if args.max_seconds:
            x = x[: int(args.max_seconds * FPS)]
        det = PourDetector(cfg)
        for i, v in enumerate(x):
            t0 = time.perf_counter()
            starts.append(t0)
            key = "start" if i == 0 else ("done" if det.update(v) == DONE else None)
            if key:
                s0 = time.perf_counter()
                tts.say(key)
                says.append(s0)
                say_cost.append(time.perf_counter() - s0)
                events.append(f"ep{ep} frame {i:4d} ({i / FPS:5.1f}s) say({key})"
                              + (f" reason={det.done_reason}" if key == "done" else ""))
            if args.work_ms:
                busy(args.work_ms)
            t_next += dt
            precise_sleep(t_next - time.perf_counter())
    t_end = time.perf_counter()
    tts.wait(timeout=10)
    tts.close()

    starts = np.array(starts + [t_end])
    period = np.diff(starts) * 1000
    t_mid = starts[:-1]
    near = np.zeros(len(period), bool)
    for s in says:
        near |= np.abs(t_mid - s) <= 1.0
    pms = dt * 1000

    lines = [
        f"TTS 루프 타이밍 테스트  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"episodes={args.episodes}  max_seconds={args.max_seconds}  work_ms={args.work_ms}  "
        f"fps={FPS}  frames={len(period)}  wall={t_end - starts[0]:.1f}s "
        f"(이상적 {len(period) * dt:.1f}s)",
        f"audio backend: {'off (--no-audio)' if args.no_audio else (tts.played and tts.played[0][1]) or 'none'}  "
        f"played={tts.played}",
        f"precise_sleep: {precise_sleep.__module__}",
        "",
        f"전체        {stats(period, pms)}",
        f"say ±1s     {stats(period[near], pms)}",
        f"그 외       {stats(period[~near], pms)}",
        f"say() 호출 소요: max={max(say_cost) * 1e3:.3f} ms  mean={np.mean(say_cost) * 1e3:.3f} ms",
        "",
        *events,
    ]
    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
