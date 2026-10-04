#!/usr/bin/env python3
"""합성 음성(edge-tts)으로 인식률 테스트 클립 만들기. 사람 녹음이 없을 때 파이프라인 점검용.

합성 음성은 발음이 또렷해서 실제 사람 목소리보다 훨씬 쉽다. 시연 전 최종 판단은
README 의 방법으로 녹음한 실제 클립으로 한다.

    pip install edge-tts   (선택 패키지. 인터넷 필요, ffmpeg 필요)
    python stt/synth_test_clips.py                  # -> <출력 폴더>/test_clips/ (기본: local/outputs/stt/test_clips)
    python stt/recognition_test.py <출력 폴더>/test_clips --device cpu

조건: quiet(원본), noisy(분홍 잡음 SNR 10dB). 잡음만 있는 none 클립도 만든다 (무음/잡음 환각 점검).
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_command import OUT_DIR, SAMPLE_RATE  # noqa: E402

VOICES = {
    "sunhi": "ko-KR-SunHiNeural",
    "injoon": "ko-KR-InJoonNeural",
    "hyunsu": "ko-KR-HyunsuMultilingualNeural",
}
PHRASES = {
    "pour": [
        "물 따라줘",
        "물 좀 따라 줄래?",
        "물 부어줘",
        "물 한 잔 줘",
        "물 좀 주세요",
        "컵에 물 좀 부어 줄래?",
        "물 따라 주세요",
        "목마른데 물 좀 줄래?",
    ],
    "stop": ["멈춰!", "그만", "정지", "스톱", "그만 따라", "잠깐, 멈춰!"],
    "none": [
        "오늘 날씨 좋네요",
        "안녕하세요, 반갑습니다",
        "이거 뭐예요?",
        "나를 따라와",
        "선물 좀 줘",
        "물건 좀 집어줘",
    ],
}
RATES = ["+0%", "+15%", "-10%"]  # 화자마다 말 빠르기를 조금씩 다르게


def pink_noise(n: int, rng) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
    x = np.fft.irfft(spec, n)
    return x / np.std(x)


def to_wav16k(mp3: Path) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def write(path: Path, x: np.ndarray) -> None:
    wavfile.write(path, SAMPLE_RATE, np.clip(x * 32768, -32768, 32767).astype(np.int16))


async def synth(text: str, voice: str, rate: str, out: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(out))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT_DIR / "test_clips")
    p.add_argument("--snr-db", type=float, default=10.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pad = np.zeros(int(0.4 * SAMPLE_RATE), np.float32)  # Enter 녹음처럼 앞뒤 여유

    with tempfile.TemporaryDirectory() as tmp:
        for vi, (vname, voice) in enumerate(VOICES.items()):
            for label, phrases in PHRASES.items():
                for i, text in enumerate(phrases):
                    mp3 = Path(tmp) / "a.mp3"
                    asyncio.run(synth(text, voice, RATES[vi], mp3))
                    x = np.concatenate([pad, to_wav16k(mp3), pad])
                    write(args.out / f"{label}__{vname}_quiet_{i:02d}.wav", x)
                    sig = np.sqrt(np.mean(x[np.abs(x) > 0.01] ** 2))  # 말소리 구간 RMS 기준
                    noisy = x + pink_noise(len(x), rng) * sig / 10 ** (args.snr_db / 20)
                    write(args.out / f"{label}__{vname}_noisy_{i:02d}.wav", noisy)
                    print(f"{label:<5} {vname:<7} {text}")
    for i in range(2):  # 잡음만 (Whisper 가 프롬프트를 환각해 pour 로 나오면 안 됨)
        write(args.out / f"none__noise_only_{i:02d}.wav", 0.02 * (i + 1) * pink_noise(2 * SAMPLE_RATE, rng))
    print(f"[저장] {args.out}  ({len(list(args.out.glob('*.wav')))}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
