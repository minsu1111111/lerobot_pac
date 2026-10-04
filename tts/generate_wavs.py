"""고정 문구 wav 사전 생성 (행사 전 1회, 인터넷 필요).

엔진: edge-tts (Microsoft 뉴럴 음성, ko-KR-SunHiNeural). 한국어 품질이 설치 가능한 후보
(gTTS, espeak-ng) 중 가장 자연스럽다. 출력은 mp3 라서 PyAV(av)로 디코딩 → wav 로 저장.

후처리: 모노 24 kHz 16-bit PCM, 앞뒤 무음 제거(앞 30 ms / 뒤 120 ms 여유), 짧은 페이드,
RMS -15 dBFS 로 (시끄러운 행사장 대비) 맞추되 피크는 -1 dBFS 를 넘지 않게.

결과 wavs/<key>.wav 는 깃에 커밋한다 (행사장 노트북은 오프라인 재생만).

다른 문장(다른 프로젝트): 문구 파일(.json 또는 .py, phrases.load_phrases 참고)과 출력 폴더를 준다.
  출력 폴더에 phrases.json(문구 목록)도 함께 써서, TTSPlayer(wav_dir=그 폴더) 만으로 자막까지 맞는다.

실행:
  python generate_wavs.py                          # 전체 (phrases.py → wavs/)
  python generate_wavs.py --keys done start        # 일부만
  python generate_wavs.py --voice ko-KR-InJoonNeural --rate +5%
  python generate_wavs.py --phrases my_phrases.json --out my_wavs
  python generate_wavs.py --voice en-US-JennyNeural --phrases en.json --out wavs_en

필요: pip install edge-tts av numpy (인터넷 필요. 재생 쪽 tts_player.py 는 필요 없음)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import wave
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from .phrases import PHRASES, load_phrases, save_manifest
except ImportError:  # 스크립트로 실행하거나 폴더를 sys.path 에 넣은 경우
    _spec = importlib.util.spec_from_file_location("_tts_phrases", HERE / "phrases.py")
    _ph = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ph)
    PHRASES, load_phrases, save_manifest = _ph.PHRASES, _ph.load_phrases, _ph.save_manifest

SR = 24000
WAV_DIR = HERE / "wavs"


async def synth_mp3(text: str, voice: str, rate: str) -> bytes:
    import edge_tts

    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if not buf:
        raise RuntimeError(f"edge-tts 가 오디오를 돌려주지 않았습니다: {text!r}")
    return bytes(buf)


def decode_mono(mp3: bytes, sr: int = SR) -> np.ndarray:
    """mp3 바이트 → float32 모노 [-1, 1], sr Hz."""
    rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
    out = []
    with av.open(io.BytesIO(mp3)) as c:
        for frame in c.decode(audio=0):
            for f in rs.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
    for f in rs.resample(None):  # 잔여 flush
        out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)


def postprocess(x: np.ndarray, sr: int = SR, thr_db: float = -40.0, rms_db: float = -15.0,
                peak_db: float = -1.0) -> np.ndarray:
    peak = np.abs(x).max()
    if peak == 0:
        return x
    # 20 ms 창 RMS 로 유성 구간 찾기 (피크 대비 thr_db)
    win = int(0.02 * sr)
    env = np.sqrt(np.convolve(x**2, np.ones(win) / win, mode="same"))
    on = np.flatnonzero(env > peak * 10 ** (thr_db / 20))
    a = max(on[0] - int(0.03 * sr), 0)
    b = min(on[-1] + int(0.12 * sr), len(x))
    x = x[a:b].copy()
    # 페이드 (클릭 방지)
    nf = int(0.005 * sr)
    x[:nf] *= np.linspace(0, 1, nf)
    x[-nf:] *= np.linspace(1, 0, nf)
    # 음량: RMS 목표, 피크 상한
    voiced = x[np.abs(x) > peak * 10 ** (thr_db / 20)]
    rms = np.sqrt(np.mean(voiced**2))
    g = min(10 ** (rms_db / 20) / rms, 10 ** (peak_db / 20) / np.abs(x).max())
    return x * g


def write_wav(path: Path, x: np.ndarray, sr: int = SR):
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description="edge-tts 로 문구 wav 생성")
    ap.add_argument("--keys", nargs="*", default=None, help="일부 키만 (기본: 전체)")
    ap.add_argument("--voice", default="ko-KR-SunHiNeural",
                    help="edge-tts 음성 (목록: edge-tts --list-voices)")
    ap.add_argument("--rate", default="+0%", help="말 속도, 예: +10%%, -5%%")
    ap.add_argument("--phrases", type=Path, default=None,
                    help="문구 파일 .json/.py (기본: phrases.py 의 PHRASES)")
    ap.add_argument("--out", type=Path, default=None,
                    help="출력 폴더 (기본: wavs/. --phrases 를 주면 반드시 지정)")
    args = ap.parse_args(argv)

    if args.phrases is not None:
        if args.out is None:
            ap.error("--phrases 를 주면 --out 도 지정하세요 (기본 wavs/ 를 덮어쓰지 않도록)")
        phrases, critical = load_phrases(args.phrases)
    else:
        phrases, critical = PHRASES, None
    out = args.out or WAV_DIR
    keys = args.keys if args.keys else list(phrases)
    unknown = [k for k in keys if k not in phrases]
    if unknown:
        ap.error(f"알 수 없는 키 {unknown} (가능: {list(phrases)})")

    out.mkdir(parents=True, exist_ok=True)
    if critical is not None:
        print(f"문구 목록: {save_manifest(out, phrases, critical)}")
    for key in keys:
        text = phrases[key]
        x = postprocess(decode_mono(asyncio.run(synth_mp3(text, args.voice, args.rate))))
        path = out / f"{key}.wav"
        write_wav(path, x)
        print(f"{key:8s} {len(x) / SR:5.2f}s  peak {20 * np.log10(np.abs(x).max()):5.1f} dBFS  "
              f"{path.stat().st_size / 1024:5.1f} KB  {text}")


if __name__ == "__main__":
    main()
