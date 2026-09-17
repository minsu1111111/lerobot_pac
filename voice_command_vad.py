#!/usr/bin/env python3
r"""음성 명령 인식 스크립트 (UNITA Manipulation / SO-101 색상 분류 프로젝트 앞단).

마이크를 듣고 있다가 말소리가 감지되면 자동으로 녹음(VAD)하고, Whisper로 한국어
텍스트 변환 후 색상 키워드를 추출해 command.json으로 저장한다.
로봇/LeRobot과는 독립적으로 동작하며, 이후 단계(색상별 정책 체크포인트 선택)는
command.json을 읽어서 처리하면 된다.

지원 OS: Ubuntu 20.04 / 22.04, Windows 10 / 11 (Python 3.10+)

파이프라인
----------
    마이크 (장치가 받는 샘플레이트/채널, int16)
      -> 16kHz mono 변환 (장치가 16kHz mono를 직접 지원하면 생략)
      -> 10/20/30ms 프레임 단위로 webrtcvad 음성 판정 -> 발화 구간만 잘라 녹음
      -> Whisper (language="ko") -> 텍스트
      -> COLOR_KEYWORDS 매칭 -> 색상 라벨 (없으면 None)
      -> command.json {"text", "color", "timestamp"}

설치 - Ubuntu 20.04 / 22.04
---------------------------
    # PortAudio 런타임(sounddevice용), webrtcvad C 확장 빌드 도구
    sudo apt install libportaudio2 build-essential
    pip install sounddevice scipy numpy openai-whisper webrtcvad

    * Python 3.10+ 필요: 22.04는 기본이 3.10, 20.04는 기본이 3.8이라 LeRobot conda 환경을
      쓰거나 deadsnakes PPA의 python3.10 + python3.10-dev + python3.10-venv 를 설치한다.
      (conda가 아닌 시스템 파이썬이면 webrtcvad 빌드에 python3.x-dev 헤더가 필요하다.)
    * webrtcvad import 시 'No module named pkg_resources' 가 나오면
      pip install "setuptools<81"  또는 pip uninstall webrtcvad && pip install webrtcvad-wheels
    * 시작할 때 ALSA/JACK 경고("Unknown PCM", "jack server is not running")가 여러 줄
      찍힐 수 있는데, PortAudio 초기화 로그일 뿐이라 동작에는 문제없다.
    * Whisper 모델 캐시: ~/.cache/whisper

설치 - Windows 10 / 11
----------------------
    pip install sounddevice scipy numpy openai-whisper webrtcvad-wheels

    * sounddevice wheel에 PortAudio가 들어 있어서 따로 설치할 것이 없다.
    * 원본 webrtcvad는 Visual C++ Build Tools가 있어야 설치되므로, 미리 빌드된
      webrtcvad-wheels를 쓴다 (API와 import 이름 webrtcvad 동일).
    * pip 기본 torch는 CPU 전용이다. GPU로 돌리려면 https://pytorch.org 에서 CUDA 빌드를 설치.
    * 설정 > 개인 정보 및 보안 > 마이크 > "데스크톱 앱이 마이크에 액세스하도록 허용"이 꺼져
      있으면 에러 없이 무음만 들어온다 (타임아웃 시 안내가 뜬다).
    * 노트북 내장 마이크 배열은 잡음 제거가 작은 소리를 0으로 지워서 말하는 도중에 녹음이
      끊길 수 있다. 설정 > 시스템 > 소리 > (입력 장치) > "오디오 향상"을 끄거나 외장/USB 마이크를 쓴다.
    * 같은 마이크가 MME / DirectSound / WASAPI 로 여러 번 보인다. 어느 것을 골라도
      16kHz로 변환해서 쓰지만, 보통은 기본값(MME)이면 충분하다.
    * Whisper 모델 캐시: %USERPROFILE%\.cache\whisper

공통
----
    * LeRobot 환경에 같이 설치해도 된다. openai-whisper는 torch 버전을 고정하지 않으므로
      이미 설치된 torch를 그대로 쓴다.
    * 녹음 데이터를 numpy 배열로 Whisper에 직접 넘기므로 ffmpeg는 필요 없다.
    * Whisper 모델은 첫 실행 시 자동 다운로드된다 (small ≈ 460MB).
    * GPU(CUDA)가 있으면 자동으로 사용하고, 없으면 CPU로 동작한다.
      로봇 정책 추론과 GPU를 나눠 쓰기 싫으면 --stt-device cpu.

사용법 (Ubuntu는 환경에 따라 python 대신 python3)
------
    python voice_command_vad.py                   # Ctrl+C 전까지 반복 인식
    python voice_command_vad.py --once            # 한 번만 인식하고 종료
    python voice_command_vad.py --model base      # 모델 크기 변경 (기본 small)
    python voice_command_vad.py --silence-ms 1500 --vad-aggressiveness 3
    python voice_command_vad.py --list-devices    # 마이크 목록 확인
    python voice_command_vad.py --mic 3           # 특정 입력 장치 사용 (번호 또는 이름 일부)

출력 (command.json, --output으로 경로 변경 가능)
    {"text": "빨간색 컵 집어줘", "color": "red", "timestamp": "2026-09-17T14:03:12+09:00"}
    색상을 못 찾으면 "color": null. 파일은 임시파일에 쓴 뒤 교체하므로
    다른 프로세스가 읽는 도중 반쯤 쓰인 파일을 보는 일은 없다.

--once 종료 코드: 0 = 색상 인식 성공, 3 = 색상 키워드 없음, 2 = 타임아웃, 1 = 오류

주의사항
--------
    * 현장이 시끄러우면 VAD가 오작동할 수 있다(잡음을 말소리로 판단해 녹음이 시작되거나,
      무음으로 판단되지 않아 녹음이 끝나지 않음). --vad-aggressiveness를 높이거나
      --silence-ms를 늘려서 튜닝이 필요하다. 녹음이 끝나지 않는 경우를 대비해
      --max-record-s(기본 15초)에서 강제로 녹음을 끊는다.
    * 안정성이 더 중요하면(시연 등) 자동 감지 대신 Enter 키로 수동 시작/종료하는
      방식을 쓰는 게 낫다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

SAMPLE_RATE = 16000  # webrtcvad, Whisper 모두 16kHz mono 기준

# 색상 라벨 -> 매칭할 한국어 표현. 색 추가/삭제는 여기만 수정하면 된다.
# 부분 문자열로 매칭하므로 "빨간"이 "빨간색", "빨간거" 등을 모두 잡는다.
COLOR_KEYWORDS: dict[str, list[str]] = {
    "red": ["빨간", "빨강", "레드", "적색"],
    "blue": ["파란", "파랑", "블루", "청색"],
    "green": ["초록", "그린", "녹색"],
    "yellow": ["노란", "노랑", "옐로"],  # "옐로"가 "옐로우"도 포함
}

# 녹음 시작 판정: 최근 START_WINDOW_MS 중 START_RATIO 이상이 음성이면 시작.
# 한 프레임만 보고 시작하면 박수/딸깍 소리에도 녹음이 켜지기 쉽다.
START_WINDOW_MS = 300
START_RATIO = 0.6
# 시작 판정 전 오디오를 이만큼 붙여서 첫 음절이 잘리지 않게 한다.
PRE_ROLL_MS = 500
# 녹음 끝의 무음은 이만큼만 남기고 잘라낸다 (Whisper가 무음 구간에서 헛소리하는 것 방지).
TRAILING_SILENCE_KEEP_MS = 300

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2
EXIT_NO_COLOR = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_console() -> None:
    # Windows 콘솔/리다이렉트(cp949)에서 인코딩 못 하는 문자가 나와도 죽지 않게 한다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# 텍스트 -> 색상 -> JSON
# --------------------------------------------------------------------------- #
def extract_color(text: str, color_keywords: dict[str, list[str]] = COLOR_KEYWORDS) -> str | None:
    """텍스트에서 색상 라벨을 찾는다. 여러 색이 나오면 가장 먼저 등장한 색을 쓴다.

    예) "빨간 컵을 파란 상자에 넣어줘" -> "red"
    공백을 제거하고 비교하므로 "빨 간색"처럼 띄어 쓴 인식 결과도 잡는다.
    """
    normalized = re.sub(r"\s+", "", text).lower()
    best: tuple[int, str] | None = None
    for label, keywords in color_keywords.items():
        for keyword in keywords:
            pos = normalized.find(keyword.lower())
            if pos != -1 and (best is None or pos < best[0]):
                best = (pos, label)
    return best[1] if best else None


def save_command(path: Path, text: str, color: str | None) -> dict:
    payload = {
        "text": text,
        "color": color,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 원자적 교체. Windows는 다른 프로세스가 파일을 열고 있는 순간 교체가 거부되므로 잠깐 재시도한다.
    retries = 20
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)
    return payload


# --------------------------------------------------------------------------- #
# 오디오 입력
# --------------------------------------------------------------------------- #
def import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("[오류] sounddevice가 없습니다: pip install sounddevice")
    except OSError as e:  # PortAudio 공유 라이브러리를 못 찾음 (주로 Linux)
        hint = "sudo apt install libportaudio2" if not IS_WINDOWS else "pip install --force-reinstall sounddevice"
        sys.exit(f"[오류] PortAudio를 불러오지 못했습니다: {e}\n       {hint}")
    return sd


def mic_help() -> str:
    tips = ["--list-devices 로 입력 장치 목록을 보고 --mic 번호로 지정하세요.",
            "마이크가 연결되어 있는지, 다른 프로그램이 독점하고 있지 않은지 확인하세요."]
    if IS_WINDOWS:
        tips += ["설정 > 개인 정보 및 보안 > 마이크 에서 '데스크톱 앱이 마이크에 액세스하도록 허용'을 켜세요.",
                 "설정 > 시스템 > 소리 > 입력 에서 장치가 '허용'이고 입력 볼륨이 0이 아닌지 확인하세요."]
    elif IS_LINUX:
        tips += ["arecord -l 로 마이크가 인식되는지, pavucontrol 에서 음소거가 아닌지 확인하세요.",
                 "'hw:' 장치가 안 열리면 --mic pulse 또는 --mic default 를 쓰세요."]
    return "\n".join(f"       - {tip}" for tip in tips)


class MicrophoneError(Exception):
    pass


class StreamResampler:
    """블록 단위 스트리밍 리샘플러 (저역통과 FIR + 선형 보간).

    블록마다 따로 리샘플링하면 경계마다 딸깍 잡음이 생기므로, 필터 상태와 보간 위치를
    블록 사이에 이어 간다. 음성 인식 입력용으로는 이 정도 품질이면 충분하다.
    """

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE):
        from scipy.signal import firwin, lfilter

        self._lfilter = lfilter
        ratio = max(in_rate / out_rate, 1.0)
        self.taps = firwin(int(32 * ratio) | 1, cutoff=0.45 * min(in_rate, out_rate), fs=in_rate)
        self.zi = np.zeros(len(self.taps) - 1)
        self.step = in_rate / out_rate
        self.pos = 0.0          # 다음 출력 샘플의 위치 (self.buf 기준, 소수)
        self.buf = np.zeros(0)  # 아직 보간에 필요한 필터링된 입력

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = self._lfilter(self.taps, 1.0, x.astype(np.float64), zi=self.zi)
        buf = np.concatenate((self.buf, y))
        last = len(buf) - 1
        if last < self.pos:
            self.buf = buf
            return np.zeros(0)
        n = int((last - self.pos) // self.step) + 1
        idx = self.pos + self.step * np.arange(n)
        out = np.interp(idx, np.arange(len(buf)), buf)
        self.pos = idx[-1] + self.step
        drop = min(int(self.pos), len(buf))
        self.buf = buf[drop:]
        self.pos -= drop
        return out


class Microphone:
    """입력 장치를 열어 16kHz mono int16 프레임(bytes)으로 넘겨준다.

    장치가 16kHz mono를 직접 못 여는 경우(Windows WASAPI, Linux ALSA 'hw:' 장치 등)에는
    장치 기본 샘플레이트/채널로 열고 소프트웨어로 다운믹스 + 리샘플링한다.
    """

    def __init__(self, sd, device: int | str | None):
        self.sd = sd
        self.device = device
        self.peak = 0  # 마지막 frames() 동안 들어온 최대 진폭 (마이크가 무음만 주는지 진단용)
        try:
            info = sd.query_devices(device, kind="input")
            hostapi = sd.query_hostapis(info["hostapi"])["name"]
            self.name = f"{info['name']} [{hostapi}]"
            self.rate, self.channels = self._pick_settings(info)
        except (sd.PortAudioError, ValueError) as e:
            raise MicrophoneError(str(e)) from e

    def _pick_settings(self, info: dict) -> tuple[int, int]:
        native_rate = int(info["default_samplerate"])
        stereo = min(2, int(info["max_input_channels"]))
        last_error: Exception | None = None
        for rate, channels in dict.fromkeys([(SAMPLE_RATE, 1), (native_rate, 1), (native_rate, stereo)]):
            try:
                self.sd.check_input_settings(device=self.device, samplerate=rate,
                                             channels=channels, dtype="int16")
                return rate, channels
            except (self.sd.PortAudioError, ValueError) as e:
                last_error = e
        raise last_error

    @property
    def needs_conversion(self) -> bool:
        return (self.rate, self.channels) != (SAMPLE_RATE, 1)

    def frames(self, frame_ms: int) -> Iterator[bytes]:
        """frame_ms 길이의 16kHz mono int16 프레임을 계속 yield한다.

        제너레이터를 close하면 스트림도 닫힌다. 녹음 한 건마다 새로 열기 때문에
        Whisper 변환 중에 쌓인 오디오가 다음 녹음으로 밀려 들어오지 않는다.
        """
        frame_len = SAMPLE_RATE * frame_ms // 1000
        blocksize = self.rate * frame_ms // 1000
        resampler = StreamResampler(self.rate) if self.rate != SAMPLE_RATE else None
        pending = np.zeros(0, dtype=np.int16)
        self.peak = 0
        warned = False

        with self.sd.RawInputStream(samplerate=self.rate, blocksize=blocksize, device=self.device,
                                    channels=self.channels, dtype="int16") as stream:
            while True:
                data, overflowed = stream.read(blocksize)
                if overflowed and not warned:
                    log("[경고] 오디오 입력 버퍼 오버플로 (일부 샘플 유실)")
                    warned = True

                raw = np.frombuffer(data, dtype=np.int16)
                if len(raw):
                    self.peak = max(self.peak, int(np.abs(raw.astype(np.int32)).max()))
                x = raw
                if self.channels > 1:
                    x = raw.reshape(-1, self.channels).mean(axis=1)
                if resampler is not None:
                    x = resampler.process(x)
                if x.dtype != np.int16:
                    x = np.clip(np.round(x), -32768, 32767).astype(np.int16)

                pending = np.concatenate((pending, x))
                while len(pending) >= frame_len:
                    yield pending[:frame_len].tobytes()
                    pending = pending[frame_len:]


# --------------------------------------------------------------------------- #
# VAD
# --------------------------------------------------------------------------- #
def import_webrtcvad():
    try:
        import webrtcvad
    except ImportError as e:
        if "pkg_resources" in str(e):  # 최신 setuptools에서 pkg_resources가 빠진 경우
            sys.exit(f"[오류] webrtcvad import 실패: {e}\n"
                     "       pip install \"setuptools<81\"  또는  pip install webrtcvad-wheels")
        if IS_WINDOWS:
            sys.exit("[오류] webrtcvad가 없습니다: pip install webrtcvad-wheels\n"
                     "       (원본 webrtcvad 패키지는 Visual C++ Build Tools가 있어야 설치됩니다)")
        sys.exit("[오류] webrtcvad가 없습니다: pip install webrtcvad\n"
                 "       (빌드 실패 시 sudo apt install build-essential python3-dev 후 재시도,\n"
                 "        또는 미리 빌드된 pip install webrtcvad-wheels)")
    return webrtcvad


def record_utterance(frames: Iterable[bytes], vad, frame_ms: int, silence_ms: int,
                     timeout_s: float, max_record_s: float, continue_vad=None) -> bytes | None:
    """말소리가 시작되면 녹음하고, silence_ms 동안 무음이면 종료한다.

    timeout_s 안에 말소리가 시작되지 않으면 None을 돌려준다.
    시작 판정은 vad(엄격)로, 녹음 중 말이 이어지는지는 continue_vad(관대)로 본다.
    엄격한 VAD로 끝까지 판정하면 작게 발음한 말끝을 무음으로 보고 말하는 도중에 끊기 쉽다.
    """
    continue_vad = continue_vad or vad
    start_frames = max(1, START_WINDOW_MS // frame_ms)
    pre_roll: deque[bytes] = deque(maxlen=max(start_frames, PRE_ROLL_MS // frame_ms))
    window: deque[bool] = deque(maxlen=start_frames)
    silence_frames = max(1, -(-silence_ms // frame_ms))  # 올림
    timeout_frames = int(timeout_s * 1000 // frame_ms)
    max_frames = int(max_record_s * 1000 // frame_ms)
    keep_tail = TRAILING_SILENCE_KEEP_MS // frame_ms

    recorded: list[bytes] = []
    triggered = False
    silence_run = 0

    for i, frame in enumerate(frames):
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        # webrtcvad는 내부적으로 배경 소음을 계속 학습하므로 녹음 전에도 매 프레임 넣어준다.
        still_speaking = continue_vad.is_speech(frame, SAMPLE_RATE)

        if not triggered:
            pre_roll.append(frame)
            window.append(is_speech)
            if len(window) == window.maxlen and sum(window) >= START_RATIO * window.maxlen:
                triggered = True
                recorded.extend(pre_roll)
                log("[녹음] 말소리 감지 - 녹음 시작")
            elif i + 1 >= timeout_frames:
                return None
            continue

        recorded.append(frame)
        silence_run = 0 if still_speaking else silence_run + 1
        if silence_run >= silence_frames:
            break
        if len(recorded) >= max_frames:
            log(f"[경고] 최대 녹음 길이 {max_record_s:g}초 도달 - 강제 종료 "
                "(주변 소음이 크면 --vad-aggressiveness를 높여보세요)")
            break

    if not triggered:  # 프레임 소스가 먼저 끝난 경우
        return None
    trim = silence_run - keep_tail
    if trim > 0:
        recorded = recorded[:-trim]
    return b"".join(recorded)


# --------------------------------------------------------------------------- #
# Whisper STT
# --------------------------------------------------------------------------- #
def load_whisper(model_name: str, stt_device: str):
    try:
        import torch
        import whisper  # 무거운 import는 실제로 변환이 필요할 때만
    except ImportError:
        sys.exit("[오류] openai-whisper가 없습니다: pip install openai-whisper")

    if stt_device == "auto":
        stt_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif stt_device == "cuda" and not torch.cuda.is_available():
        hint = ("\n       Windows의 pip 기본 torch는 CPU 전용입니다. GPU를 쓰려면 https://pytorch.org 에서 CUDA 빌드를 설치하세요."
                if IS_WINDOWS else "")
        sys.exit(f"[오류] --stt-device cuda 를 지정했지만 CUDA를 사용할 수 없습니다. --stt-device cpu 로 실행하세요.{hint}")

    log(f"[STT] Whisper '{model_name}' 모델 로딩 중 ({stt_device}) - 첫 실행이면 다운로드에 시간이 걸립니다...")
    try:
        model = whisper.load_model(model_name, device=stt_device)
    except RuntimeError as e:  # 잘못된 모델 이름, GPU 메모리 부족 등
        sys.exit(f"[오류] Whisper 모델 로딩 실패: {e}\n"
                 "       GPU 메모리 부족이면 --stt-device cpu 또는 더 작은 --model 을 쓰세요.")
    return model, stt_device


def transcribe(model, pcm: bytes, stt_device: str) -> str:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(
        audio,
        language="ko",
        task="transcribe",
        fp16=(stt_device == "cuda"),  # CPU는 fp16 미지원 (경고 방지)
        condition_on_previous_text=False,
    )
    return result["text"].strip()


def save_wav(path: Path, pcm: bytes) -> None:
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SAMPLE_RATE, np.frombuffer(pcm, dtype=np.int16))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VAD 기반 음성 명령 인식: 음성 -> 텍스트(Whisper, ko) -> 색상 -> command.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="small",
                   help="Whisper 모델 크기 (tiny/base/small/medium/large/turbo)")
    p.add_argument("--once", action="store_true", help="한 번만 인식하고 종료 (기본: Ctrl+C 전까지 반복)")
    p.add_argument("--silence-ms", type=int, default=1200,
                   help="이 시간(ms) 동안 무음이면 녹음 종료. 말하다 끊기면 늘리세요")
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=range(4),
                   help="VAD 민감도 0(관대)~3(엄격). 녹음 시작 판정에 쓰고, 녹음 중에는 한 단계 관대하게 판정. "
                        "시끄러우면 높이세요")
    p.add_argument("--timeout", type=float, default=15.0, help="말소리가 없을 때 대기 시간(초)")
    p.add_argument("--frame-ms", type=int, default=30, choices=(10, 20, 30), help="VAD 판정 프레임 길이(ms)")
    p.add_argument("--max-record-s", type=float, default=15.0, help="최대 녹음 길이(초)")
    p.add_argument("--mic", type=parse_device, default=None, help="입력 장치 번호 또는 이름 일부 (기본: 시스템 기본 마이크)")
    p.add_argument("--list-devices", action="store_true", help="오디오 장치 목록 출력 후 종료")
    p.add_argument("--stt-device", default="auto", choices=("auto", "cpu", "cuda"), help="Whisper 실행 장치")
    p.add_argument("--output", type=Path, default=Path("command.json"), help="결과 JSON 경로")
    p.add_argument("--save-wav", type=Path, default=None, help="마지막 녹음을 wav로 저장 (VAD 튜닝/디버깅용)")
    args = p.parse_args(argv)
    if args.silence_ms <= 0 or args.timeout <= 0 or args.max_record_s <= 0:
        p.error("--silence-ms, --timeout, --max-record-s 는 0보다 커야 합니다")
    return args


def listen_once(args: argparse.Namespace, mic: Microphone, vad, continue_vad, model, stt_device: str) -> int:
    log(f"\n[대기] 말씀하세요... ({args.timeout:g}초 동안 말이 없으면 타임아웃)")
    with closing(mic.frames(args.frame_ms)) as frames:
        pcm = record_utterance(frames, vad, args.frame_ms, args.silence_ms,
                               args.timeout, args.max_record_s, continue_vad)
    if pcm is None:
        log(f"[타임아웃] {args.timeout:g}초 동안 말소리가 감지되지 않았습니다.")
        if mic.peak == 0:
            log("[안내] 마이크 입력이 완전히 0(무음)입니다. 음소거되었거나 마이크 권한이 막혀 있을 수 있습니다.\n"
                + mic_help())
        return EXIT_TIMEOUT

    duration = len(pcm) / 2 / SAMPLE_RATE
    log(f"[녹음] 종료 ({duration:.1f}초) - 텍스트 변환 중...")
    if args.save_wav:
        save_wav(args.save_wav, pcm)

    text = transcribe(model, pcm, stt_device)
    log(f"[인식] \"{text}\"")
    color = extract_color(text)
    if color is None:
        log(f"[안내] 색상 키워드를 찾지 못했습니다. 인식 가능한 색: {', '.join(COLOR_KEYWORDS)}")
    else:
        log(f"[색상] {color}")

    try:
        save_command(args.output, text, color)
    except OSError as e:
        log(f"[오류] {args.output} 저장 실패: {e}")
        return EXIT_ERROR
    log(f"[저장] {args.output}")
    return EXIT_OK if color else EXIT_NO_COLOR


def main(argv: list[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    sd = import_sounddevice()

    if args.list_devices:
        print(sd.query_devices())
        if IS_WINDOWS:
            print("\n* 같은 마이크가 MME / DirectSound / WASAPI 로 중복 표시됩니다. 입력 채널(in)이 있는 장치 번호를 --mic 로 지정하세요.")
        return EXIT_OK

    webrtcvad = import_webrtcvad()
    vad = webrtcvad.Vad(args.vad_aggressiveness)                      # 녹음 시작: 잡음에 안 켜지게 엄격히
    continue_vad = webrtcvad.Vad(max(0, args.vad_aggressiveness - 1))  # 녹음 유지: 작은 말끝도 놓치지 않게
    try:  # 모델 로딩 전에 마이크 문제부터 빨리 알려준다
        mic = Microphone(sd, args.mic)
    except MicrophoneError as e:
        target = "기본 입력 장치(마이크)" if args.mic is None else f"입력 장치 {args.mic!r}"
        sys.exit(f"[오류] {target}를 열 수 없습니다: {e}\n{mic_help()}")
    conversion = f" ({mic.rate}Hz {mic.channels}ch로 열고 16kHz mono로 변환)" if mic.needs_conversion else ""
    log(f"[마이크] {mic.name}{conversion}")

    try:
        model, stt_device = load_whisper(args.model, args.stt_device)
        while True:
            code = listen_once(args, mic, vad, continue_vad, model, stt_device)
            if args.once:
                return code
    except KeyboardInterrupt:
        log("\n[종료] Ctrl+C")
        return EXIT_OK
    except sd.PortAudioError as e:
        log(f"[오류] 오디오 입력 중 문제가 발생했습니다 (마이크 연결 해제?): {e}\n{mic_help()}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
