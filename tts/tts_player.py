"""TTS 재생기. 제어 루프(30fps)를 막지 않도록 별도 스레드에서 미리 만든 wav 를 튼다.

  say(key)  : 즉시 반환(자막 문자열). 재생은 큐에 넣고 워커 스레드가 순서대로 재생.
  wait()    : 큐가 빌 때까지 대기 (종료 직전 '다 따랐습니다' 끝까지 듣기 등)
  close()   : 워커 종료, 재생 중이면 끊음

큐 정책: 끊지 않고 줄 세운다. 같은 키가 이미 대기 중이면 무시(중복 안내 방지).
  대기열이 max_pending 을 넘으면 오래된 일반 문구부터 버리고, CRITICAL_KEYS(done 등)는 버리지 않는다.

재생 백엔드(위에서부터 시도, 실패하면 다음): aplay(alsa-utils) → paplay(PulseAudio).
  (pw-play 는 22.04 버전이 표준입력 재생을 못 해서 뺐다.)
  wav 는 init 때 메모리에 올리고 raw PCM 을 표준입력으로 흘려 넣는다(재생 중 디스크 접근 없음).
  모두 실패하면 한 번만 경고하고 자막만 계속한다 — 제어 루프로 예외를 올리지 않는다.
  Ubuntu 20.04 데스크톱 기본 설치에 aplay, paplay 가 둘 다 있다.
  backends 에는 명령 이름 대신 함수 fn(pcm: bytes, sr: int, device) 도 넣을 수 있다
  (예외를 던지면 실패로 보고 다음 백엔드로). 테스트용 가짜 백엔드, 다른 오디오 라이브러리 연결용.

독립 폴더: 이 폴더(tts/)만 복사해도 동작한다. 표준 라이브러리만 쓴다 (프로젝트 paths.py 등 불필요).
  문구/wav 를 바꾸려면 TTSPlayer(wav_dir=..., phrases=...) — README 참고.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WAV_DIR = HERE / "wavs"

try:  # 패키지로 import (from tts.tts_player import ...) 한 경우
    from .phrases import CRITICAL_KEYS, MANIFEST, PHRASES, load_phrases
except ImportError:  # 폴더를 sys.path 에 넣고 쓰는 경우. 이름이 같은 다른 'phrases' 모듈과 섞이지 않게 파일로 로드
    _spec = importlib.util.spec_from_file_location("_tts_phrases", HERE / "phrases.py")
    _ph = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ph)
    CRITICAL_KEYS, MANIFEST, PHRASES, load_phrases = (
        _ph.CRITICAL_KEYS, _ph.MANIFEST, _ph.PHRASES, _ph.load_phrases)


def output_dir() -> Path:
    """결과물 폴더 (만들지는 않음).
    $UNITA_LOCAL/outputs/tts → (../../local 또는 ~/UNITA_PAC2026/local 이 있으면) 그 아래 outputs/tts → tts/outputs."""
    if os.environ.get("UNITA_LOCAL"):
        return Path(os.environ["UNITA_LOCAL"]) / "outputs" / "tts"
    for local in ([HERE.parents[1] / "local"] if len(HERE.parents) > 1 else []) + [Path.home() / "UNITA_PAC2026" / "local"]:
        if local.is_dir():
            return local / "outputs" / "tts"
    return HERE / "outputs"


@dataclass
class Clip:
    key: str
    text: str
    pcm: bytes  # s16le mono
    sr: int

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 2 / self.sr


def _cmd(backend: str, sr: int, device: str | None) -> list[str]:
    if backend == "aplay":
        cmd = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(sr), "-c", "1"]
        return cmd + (["-D", device] if device else []) + ["-"]
    if backend == "paplay":
        cmd = ["paplay", "--raw", "--format=s16le", f"--rate={sr}", "--channels=1"]
        return cmd + ([f"--device={device}"] if device else [])
    raise ValueError(backend)


BACKENDS = ("aplay", "paplay")


def _name(backend) -> str:
    return backend if isinstance(backend, str) else getattr(backend, "__name__", repr(backend))


def resolve_phrases(wav_dir: Path, phrases=None, critical_keys=None
                    ) -> tuple[dict[str, str], frozenset[str]]:
    """phrases: None → wav_dir/phrases.json 이 있으면 그것, 없으면 기본 PHRASES.
    str/Path → 문구 파일(.json/.py). Mapping → 그대로. critical_keys 를 주면 그것이 우선."""
    if phrases is None:
        manifest = Path(wav_dir) / MANIFEST
        phrases, critical = load_phrases(manifest) if manifest.is_file() else (PHRASES, CRITICAL_KEYS)
    elif isinstance(phrases, (str, Path)):
        phrases, critical = load_phrases(phrases)
    elif isinstance(phrases, Mapping):
        critical = CRITICAL_KEYS
    else:
        raise TypeError(f"phrases 는 None, 파일 경로, 또는 {{키: 문장}} 이어야 합니다: {type(phrases)}")
    if critical_keys is not None:
        critical = critical_keys
    return dict(phrases), frozenset(critical)


class TTSPlayer:
    def __init__(self, wav_dir: str | Path | None = None, device: str | None = None,
                 enabled: bool = True, max_pending: int = 3, backends: Iterable = BACKENDS,
                 phrases: Mapping[str, str] | str | Path | None = None,
                 critical_keys: Iterable[str] | None = None):
        """wav_dir   : <key>.wav 폴더 (기본: 이 폴더의 wavs/)
        device    : aplay -D / paplay --device 값 (None 이면 시스템 기본 장치)
        enabled   : False 면 소리 없이 자막(반환 문자열·print)만
        max_pending: 대기열 길이 상한 (넘치면 오래된 일반 문구부터 버림)
        backends  : 시도 순서. 명령 이름(str) 또는 함수 fn(pcm, sr, device)
        phrases   : 자막 문구 {key: text} 또는 문구 파일 경로 (기본: resolve_phrases 참고)
        critical_keys: 버리지 않을 키 (기본: 문구 파일의 critical, 없으면 CRITICAL_KEYS)"""
        wav_dir = Path(wav_dir) if wav_dir is not None else DEFAULT_WAV_DIR
        self.phrases, self.critical_keys = resolve_phrases(wav_dir, phrases, critical_keys)
        self.clips = self._load(wav_dir, self.phrases)
        self.device = device
        self.enabled = enabled
        self.max_pending = max_pending
        backends = list(backends)
        self.backends = [b for b in backends if callable(b) or shutil.which(b)]
        self.backend: str | None = _name(self.backends[0]) if self.backends else None  # 현재 쓰는 백엔드
        self.played: list[tuple[str, str]] = []  # (key, backend) 실제 재생 성공 기록
        self._q: deque[str] = deque()
        self._cv = threading.Condition()
        self._busy = False
        self._stop = False
        self._proc: subprocess.Popen | None = None
        self._muted_logged = False
        if enabled and not self.backends:
            self._mute(f"재생 프로그램 없음 (후보 {[_name(b) for b in backends]})")
        self._th = threading.Thread(target=self._worker, name="tts", daemon=True)
        self._th.start()

    # ---- 공개 API (제어 루프 스레드에서 호출) ----
    def say(self, key: str) -> str:
        """비차단. 자막 문자열을 돌려준다. 알 수 없는 키는 경고만 하고 빈 문자열."""
        clip = self.clips.get(key)
        if clip is None:
            print(f"[TTS] 알 수 없는 키 {key!r} (가능: {list(self.clips)})", flush=True)
            return ""
        print(f"[TTS] {clip.text}", flush=True)
        if not self.enabled:
            return clip.text
        with self._cv:
            if key not in self._q:
                self._q.append(key)
                while len(self._q) > self.max_pending:
                    drop = next((k for k in self._q if k not in self.critical_keys), None)
                    if drop is None:
                        break
                    self._q.remove(drop)
            self._cv.notify()
        return clip.text

    def wait(self, timeout: float | None = None) -> bool:
        """대기열과 현재 재생이 끝날 때까지 기다린다. 시간 안에 끝나면 True."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._q or self._busy:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self):
        with self._cv:
            self._stop = True
            self._q.clear()
            self._cv.notify_all()
        p = self._proc
        if p is not None and p.poll() is None:
            p.kill()
        self._th.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 내부 ----
    @staticmethod
    def _load(wav_dir: Path, phrases: Mapping[str, str]) -> dict[str, Clip]:
        missing = [k for k in phrases if not (wav_dir / f"{k}.wav").is_file()]
        if missing:
            raise FileNotFoundError(
                f"TTS wav 파일이 없습니다: {missing} (폴더: {wav_dir}). "
                f"인터넷 되는 곳에서 'python generate_wavs.py' (문구를 바꿨으면 --phrases 파일 --out 폴더) "
                f"로 만든 뒤 커밋하세요.")
        clips = {}
        for k, text in phrases.items():
            with wave.open(str(wav_dir / f"{k}.wav"), "rb") as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    raise ValueError(f"{k}.wav 는 16-bit 모노여야 합니다 "
                                     f"(현재 {8 * w.getsampwidth()}-bit, {w.getnchannels()}ch)")
                clips[k] = Clip(k, text, w.readframes(w.getnframes()), w.getframerate())
        return clips

    def _mute(self, why: str):
        self.backend = None
        if not self._muted_logged:
            self._muted_logged = True
            print(f"[TTS] 소리 재생 불가 → 자막만 표시합니다. 이유: {why}", flush=True)

    def _play(self, clip: Clip):
        """현재 백엔드로 재생. 실패하면 다음 백엔드로 넘어가 같은 문구를 다시 시도."""
        while self.backends:
            b = self.backends[0]
            self.backend = _name(b)
            if callable(b):
                try:
                    b(clip.pcm, clip.sr, self.device)
                    rc = 0
                except Exception as e:
                    rc, err = -1, repr(e)
            else:
                rc, err = self._play_cmd(b, clip)
            if rc == 0:
                self.played.append((clip.key, _name(b)))
                return
            if self._stop:
                return
            print(f"[TTS] {_name(b)} 재생 실패 (rc={rc}): {err.splitlines()[-1] if err else ''}",
                  flush=True)
            self.backends.pop(0)
        self._mute("모든 백엔드 실패 (오디오 장치 없음?)")

    def _play_cmd(self, b: str, clip: Clip) -> tuple[int, str]:
        err = ""
        try:
            p = self._proc = subprocess.Popen(
                _cmd(b, clip.sr, self.device), stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                p.stdin.write(clip.pcm)
                p.stdin.close()
            except BrokenPipeError:
                pass
            err = p.stderr.read().decode(errors="replace").strip()
            rc = p.wait()
        except OSError as e:
            rc, err = -1, str(e)
        finally:
            self._proc = None
        return rc, err

    def _worker(self):
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                key = self._q.popleft()
                self._busy = True
            try:
                if self.backends:
                    self._play(self.clips[key])
            except Exception as e:  # 어떤 경우에도 워커가 죽지 않게
                print(f"[TTS] 재생 중 예외: {e!r}", flush=True)
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="TTS 문구 재생 확인")
    ap.add_argument("keys", nargs="*", help="재생할 키 (없으면 전체)")
    ap.add_argument("--wav-dir", type=Path, default=None, help="wav 폴더 (기본: wavs/)")
    ap.add_argument("--phrases", type=Path, default=None, help="문구 파일 .json/.py")
    ap.add_argument("--device", default=None, help="aplay -D / paplay --device 값")
    ap.add_argument("--no-audio", action="store_true", help="소리 없이 자막만")
    args = ap.parse_args(argv)

    probe = TTSPlayer(wav_dir=args.wav_dir, phrases=args.phrases, enabled=False)
    probe.close()
    keys = args.keys or list(probe.phrases)
    with TTSPlayer(wav_dir=args.wav_dir, phrases=args.phrases, device=args.device,
                   enabled=not args.no_audio, max_pending=max(len(keys), 1)) as tts:
        print("backends:", [_name(b) for b in tts.backends])
        for k in keys:
            tts.say(k)
        tts.wait()
        print("played:", tts.played)


if __name__ == "__main__":
    # 사용 예: python tts_player.py start done   (인자 없으면 전체)
    main()
