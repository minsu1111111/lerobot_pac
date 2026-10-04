"""시뮬 롤아웃 영상(mp4)에 TTS 소리를 입힌다.

pour_rollout.py 의 run_XX.csv 에서 `tts:<key>` 이벤트 step 을 읽어 같은 시점에 wav 를 깐다.
영상 프레임 = 붓기 1회마다 리셋 1장 + step 당 1장 (MujocoBiSO101) 이므로 step s → 프레임 s+1.
여러 번 부은 세션이면 run 순서대로 프레임 수를 누적한다. 겹치는 안내는 TTSPlayer 처럼 큐로 이어 붙인다.

실행:
  python mux_tts_audio.py VIDEO.mp4 RUN_DIR [-o OUT.mp4]
pour_rollout.py --sim-video 로 찍으면 끝날 때 자동으로 불린다 (<이름>_audio.mp4).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import GITHUB  # noqa: E402

FPS = 30
WAV_DIR = GITHUB / "tts" / "wavs"


def tts_events(run_dir: Path) -> list[tuple[float, str]]:
    """(영상 시각 s, key) 목록."""
    out, frame0 = [], 0
    for csv in sorted(run_dir.glob("run_*.csv")):
        r = pd.read_csv(csv)
        for step, ev in zip(r["step"], r["event"].fillna("")):
            for e in str(ev).split(";"):
                if e.startswith("tts:"):
                    out.append(((frame0 + 1 + int(step)) / FPS, e[4:]))
        frame0 += 1 + int(r["step"].max()) + 1
    return out


def build_track(events, duration_s: float, wav_dir: Path = WAV_DIR):
    clips = {}
    sr = None
    for _, key in events:
        if key not in clips:
            rate, x = wavfile.read(wav_dir / f"{key}.wav")
            sr = sr or rate
            assert rate == sr, "wav 샘플레이트가 서로 다름"
            clips[key] = x.astype(np.float32) / 32768.0
    sr = sr or 24000
    track = np.zeros(int(duration_s * sr) + 1, np.float32)
    busy_until = 0  # 큐: 앞 안내가 끝난 뒤 시작
    for t, key in events:
        x = clips[key]
        i = max(int(t * sr), busy_until)
        j = min(i + len(x), len(track))
        if i < j:
            track[i:j] += x[: j - i]
        busy_until = i + len(x)
    return sr, np.clip(track, -1, 1)


def mux(video: Path, run_dir: Path, out: Path | None = None) -> Path:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = out or video.with_name(video.stem + "_audio.mp4")
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr
    n_frames = int(probe.replace("\r", "\n").split("frame=")[-1].split()[0])
    events = tts_events(run_dir)
    sr, track = build_track(events, n_frames / FPS)
    wav = out.with_suffix(".tts.wav")
    wavfile.write(wav, sr, (track * 32767).astype(np.int16))
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out),
        ],
        check=True,
    )
    wav.unlink()
    print(f"[영상] TTS {len(events)}개 ({', '.join(f'{k}@{t:.1f}s' for t, k in events)}) → {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("run_dir", type=Path, help="pour_rollout 결과 폴더 (run_XX.csv)")
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()
    mux(a.video, a.run_dir, a.out)


if __name__ == "__main__":
    main()
