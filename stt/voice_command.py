#!/usr/bin/env python3
"""음성 명령 인식 (PAC 2026 "말하면 따라주는 손" 앞단).

이전 색 분류 프로젝트의 voice_command_vad.py(lerobot_pac)를 가져와 명령어 인식용으로 고쳤다.
마이크 -> 16kHz mono -> 녹음 -> Whisper(ko) -> 명령어 매칭(pour/stop) -> CommandResult.

녹음 방식
  enter (기본): Enter 로 녹음 시작, Enter 로 끝 (최대 --max-record-s 초). 시연용으로 안정적.
                녹음 대신 키를 치면 같은 명령으로 처리 (p=따라줘, s=멈춰, q=종료).
  vad         : 말소리가 감지되면 자동 녹음 (webrtcvad). 현장이 조용할 때만 권장.

이 폴더(stt/)는 다른 폴더를 import 하지 않는다. 폴더째 복사해서 다른 프로젝트에 넣어도 동작한다.

라이브러리 (rollout 에서 사용)
    from voice_command import VoiceCommander       # stt/ 를 sys.path 에 넣었을 때
    # 또는 from stt import VoiceCommander          # stt/ 의 부모 폴더가 sys.path 에 있을 때
    vc = VoiceCommander(model="small", device="cpu")   # Whisper 는 여기서 한 번만 로딩
    r = vc.listen()           # CommandResult(text, command, source, duration_s, stt_s)
    if r.command == "pour": ...
    vc.poll_keyboard()        # 로봇 동작 중 non-blocking 으로 s+Enter(정지) 확인

CLI
    python stt/voice_command.py                  # Enter 모드 반복 (q 또는 Ctrl+C 로 종료)
    python stt/voice_command.py --once           # 한 번만
    python stt/voice_command.py --mode vad       # 자동 감지 모드
    python stt/voice_command.py --wav a.wav b.wav  # 마이크 대신 wav 파일 인식
    python stt/voice_command.py --list-devices

출력: <출력 폴더>/command.json {"text", "command", "source", "timestamp"} (원자적 교체)
  출력 폴더 = $UNITA_LOCAL/outputs/stt  >  (stt/../../local 또는 ~/UNITA_PAC2026/local 이 있으면) 그 아래 outputs/stt  >  stt/outputs
--once 종료 코드: 0 = 명령 인식(또는 q), 3 = 명령 없음, 2 = 타임아웃(vad), 1 = 오류

설치: pip install -r stt/requirements.txt   (torch 는 사용 중인 환경의 것을 그대로 씀)
  Ubuntu: sudo apt install libportaudio2  (없으면 sounddevice import 시 'PortAudio library not found')
  webrtcvad 의 pkg_resources 경고/오류는 pip install "setuptools<81".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

SAMPLE_RATE = 16000  # webrtcvad, Whisper 모두 16kHz mono 기준


def default_out_dir(module_file: str | Path = __file__) -> Path:
    """결과물(command.json, 인식률 CSV, 녹음 클립) 기본 폴더. 다른 폴더를 import 하지 않고 정한다.

    1) 환경변수 UNITA_LOCAL 이 있으면  $UNITA_LOCAL/outputs/stt
    2) 이 파일 기준 ../../local 또는 ~/UNITA_PAC2026/local 이 있으면 그 아래 outputs/stt
    3) 그 밖에 (폴더만 복사해 간 경우)  stt/outputs   (.gitignore 에 들어 있음)
    """
    env = os.environ.get("UNITA_LOCAL")
    if env:
        return Path(env).expanduser() / "outputs" / "stt"
    here = Path(module_file).resolve()
    parents = here.parents
    for local in ([parents[2] / "local"] if len(parents) > 2 else []) + [
        Path.home() / "UNITA_PAC2026" / "local"
    ]:
        if local.is_dir():
            return local / "outputs" / "stt"
    return here.parent / "outputs"


OUT_DIR = default_out_dir()

# --------------------------------------------------------------------------- #
# 명령어 사전. 공백·문장부호를 지운 텍스트에서 부분 문자열로 찾는다.
# --------------------------------------------------------------------------- #
POUR, STOP, QUIT = "pour", "stop", "quit"
BACK, RESUME = "back", "resume"
HOME = "home"  # 대기 중 "초기자세로" → 팔을 프로그램 시작 때 자세로  # 붓는 중 정지한 뒤: 되감아 돌아가기 / 이어서 하기

# 팀이 정한 기본 명령: "따라줘" / "정지". 나머지는 인식 실패 대비 동의어.
COMMANDS: dict[str, list[str]] = {
    # "~지 마"(따르지 마) 도 안전하게 정지로 본다.
    # "정진"/"스토"/"스톰"/"스프" 는 Whisper 가 "정지"/"스톱"을 잘못 받아쓴 형태 (합성 음성 시험에서 확인).
    # "정치" 는 붓는 중 음성 정지(서보 소음)에서 "정지"를 받아쓴 형태 (follower1 실측).
    # stop 쪽 오탐은 안전하므로(로봇이 멈출 뿐) 오인식 형태를 넉넉히 받는다.
    STOP: [
        "정지",
        "정진",
        "정치",
        "멈춰",
        "멈추",
        "멈춤",
        "그만",
        "중지",
        "스톱",
        "스탑",
        "스토",
        "스톰",
        "스프",
        "stop",
        "멈출",
        "잠깐",
        "잠시",
        "기다려",
        "지마",
        "지말",
        # 뜻이 같은 말 (붓는 중 "됐어"/"충분해" = 그만, "안 돼"/"위험해" = 멈춰). 오탐해도 멈출 뿐.
        "됐어",
        "충분",
        "안돼",
        "안되",
        "위험",
    ],
    # 대기 중 "초기자세로" → 처음 자세로 복귀 (일시정지 중이면 "돌아가"와 같게 처리)
    HOME: [
        "초기자세",
        "처음자세",
        "기본자세",
        "원래자세",
        "초기상태",
        "처음상태",
        "기본상태",
        "원래상태",
        "초기화",
        "홈으로",
        "원점",
    ],
    # 정지 뒤 "돌아가" → 되감기 복귀 (rollout 일시정지에서만 쓰고, 대기 중엔 무시)
    BACK: [
        "돌아가",
        "돌아와",
        "되돌아",
        "되돌려",
        "원래대로",
        "원래자리",
        "원위치",
        "원상태",
        "원상복구",
        "제자리",
        "처음으로",
        "처음자리",
        "복귀",
        "취소",
        "내려놔",
        "내려놓",
        "안할래",
        "필요없",
    ],
    # 정지 뒤 "계속 해줘" → 이어서 붓기 (일시정지 중엔 "따라줘" 도 계속으로 본다)
    # "맞아해" = 스피커→마이크에서 "마저 해줘"를 받아쓴 형태 (실측). "진행해"는 실측 0/3 ("지냉이")이라 기대하지 말 것.
    RESUME: [
        "계속",
        "이어서",
        "이어해",
        "마저",
        "맞아해",
        "진행",
        "다시해",
        "다시시작",
        "괜찮아",
        "괜찮으니",
        "더해",
        "고고",
    ],
    # 붓는 동사. "물" 없이 "따라줘"만 말해도 pour.
    # "목말라"/"채워줘"/"담아줘" 처럼 물을 달라는 다른 말도 시작.
    POUR: ["따라", "따뤄", "따르", "부어", "한잔", "채워", "담아", "목말라", "목이말라", "갈증"],
}
# 매칭 전에 지우는 표현 (다른 뜻의 "따라"/"따르").
COMMAND_EXCLUDE: dict[str, list[str]] = {
    STOP: ["스토리", "스토어", "스토브", "스토킹"],
    # "물 따르는 건 잘해"처럼 설명하는 말(대회장 옆 대화에서 실제로 시작이 걸림)은 시작으로 치지 않는다
    POUR: [
        "따라와",
        "따라가",
        "따라온",
        "따라간",
        "따라서",
        "따라해",
        "따라하",
        "따라잡",
        "따르면",
        "따른",
        "따르는",
        "따랐",
        "따르고",
        "따르다",
    ],
}
# 둘 다 나오면 앞쪽이 이긴다 (안전: stop 우선).
COMMAND_PRIORITY: tuple[str, ...] = (STOP, HOME, BACK, POUR, RESUME)
# 붓는 중 정지한 뒤(일시정지)에는 "그만 돌아가" 처럼 정지어가 섞여도 고르는 말이 이긴다 (이미 멈춰 있음).
PAUSE_PRIORITY: tuple[str, ...] = (HOME, BACK, POUR, RESUME, STOP)

# "물" + 요청 표현 조합도 pour ("물 좀 줘", "물 한 잔 주세요", "물 마시고 싶어").
# 선물/동물/물건/물어봐 같은 단어의 "물"은 제외한다.
WATER_RE = re.compile(r"(?<![선동식생건괴보약인산폐광])물(?![건어었론고기리질체품감결러])")
WATER_REQUEST_RE = re.compile(r"줘|주세요|주시|주라|줄래|줄수|좀|잔|마시|마실|원해|필요|부탁|채워")

# Whisper 가 명령어 어휘 쪽으로 받아쓰도록 주는 힌트. 무음에서 프롬프트를 그대로 뱉는
# 환각이 나와도 stop 어휘가 함께 들어 있어 stop 이 이기도록 둘 다 넣는다.
INITIAL_PROMPT = "물 따라줘. 정지."

# Whisper 디코딩 상한 (토큰). 명령어는 5~15 토큰이면 충분하다. 기본값(224)이면 잡음에서 같은 말을
# 반복하는 환각이 나올 때 CPU 로 토큰당 60~120ms x 224 = 15~27초가 걸릴 수 있어 48로 묶는다.
SAMPLE_LEN = 48

# Enter 모드 키보드 대체 입력. 한글 자판 상태에서 친 경우(ㅔ/ㄴ/ㅂ)도 받는다.
KEY_COMMANDS: dict[str, str] = {"p": POUR, "ㅔ": POUR, "s": STOP, "ㄴ": STOP, "h": HOME, "ㅗ": HOME}
QUIT_KEYS = {"q", "ㅂ", "quit", "exit"}
KEY_HINT = "키보드로 대신 입력: p+Enter = 따라줘, s+Enter = 정지, q+Enter = 종료"

# VAD 모드 녹음 판정 (원본 그대로)
START_WINDOW_MS = 300  # 최근 300ms 중
START_RATIO = 0.6  # 60% 이상 음성이면 녹음 시작 (딸깍 소리에 안 켜지게)
PRE_ROLL_MS = 500  # 시작 판정 전 오디오를 붙여 첫 음절 보존
TRAILING_SILENCE_KEEP_MS = 300  # 끝 무음은 이만큼만 남김 (무음 환각 방지)
# Enter 모드: 두 번째 Enter 뒤에도 조금 더 녹음 (말끝과 동시에 누르는 경우)
POST_ROLL_MS = 200
# 녹음 최대 진폭이 이보다 작으면 무음으로 보고 Whisper 를 건너뛴다 (무음 환각 방지)
MIN_PEAK = 300

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2
EXIT_NO_COMMAND = 3


def log(msg: str) -> None:
    print(msg, flush=True)


def configure_console() -> None:
    # Windows 콘솔/리다이렉트(cp949)에서 인코딩 못 하는 문자가 나와도 죽지 않게 한다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


class SttError(RuntimeError):
    """설치/장치/모델 문제. 메시지에 한국어 해결 방법이 들어 있다."""


@dataclass
class CommandResult:
    text: str  # 인식 텍스트 (키보드면 친 문자열)
    command: str | None  # "pour" | "stop" | "quit" | None
    source: str  # "voice" | "keyboard"
    duration_s: float = 0.0  # 녹음 길이
    stt_s: float = 0.0  # Whisper 변환 시간


# --------------------------------------------------------------------------- #
# 텍스트 -> 명령 -> JSON
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """공백·문장부호 제거 + 소문자. "물 좀 따라 줄래?" -> "물좀따라줄래"."""
    return re.sub(r"[\s\W_]+", "", text).lower()


def match_command(
    text: str,
    commands: dict[str, list[str]] = COMMANDS,
    exclude: dict[str, list[str]] = COMMAND_EXCLUDE,
    priority: tuple[str, ...] = COMMAND_PRIORITY,
) -> str | None:
    """텍스트에서 명령을 찾는다 (순수 함수). 여러 개면 priority 순서(stop 우선).

    예) "물 따라줘" -> "pour", "그만 따라" -> "stop", "나를 따라와" -> None
    """
    norm = normalize(text)
    for cmd in sorted(commands, key=lambda c: priority.index(c) if c in priority else len(priority)):
        t = norm
        for ex in exclude.get(cmd, []):
            t = t.replace(ex, "")
        if any(k.lower() in t for k in commands[cmd]):
            return cmd
        if cmd == POUR and WATER_RE.search(t) and WATER_REQUEST_RE.search(t):
            return cmd
    return None


def parse_enter_input(line: str) -> tuple[str, str | None]:
    """Enter 모드에서 친 한 줄을 해석한다.

    ""(Enter만)      -> ("record", None)
    p / s / ㅔ / ㄴ   -> ("command", "pour"/"stop")
    q / ㅂ / quit    -> ("quit", "quit")
    그 밖의 문장      -> 명령어 매칭 ("물 따라줘" 타이핑도 됨), 없으면 ("unknown", None)
    """
    s = line.strip().lower()
    if not s:
        return "record", None
    if s in QUIT_KEYS:
        return "quit", QUIT
    if s in KEY_COMMANDS:
        return "command", KEY_COMMANDS[s]
    cmd = match_command(s)
    return ("command", cmd) if cmd else ("unknown", None)


def save_command(path: Path, result: CommandResult) -> dict:
    payload = {
        "text": result.text,
        "command": result.command,
        "source": result.source,
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
    except ImportError as e:
        raise SttError("[오류] sounddevice가 없습니다: pip install sounddevice") from e
    except OSError as e:  # PortAudio 공유 라이브러리를 못 찾음 (주로 Linux)
        hint = (
            "sudo apt install libportaudio2"
            if not IS_WINDOWS
            else "pip install --force-reinstall sounddevice"
        )
        raise SttError(f"[오류] PortAudio를 불러오지 못했습니다: {e}\n       {hint}") from e
    return sd


def mic_help() -> str:
    tips = [
        "--list-devices 로 입력 장치 목록을 보고 --mic 번호로 지정하세요.",
        "마이크가 연결되어 있는지, 다른 프로그램이 독점하고 있지 않은지 확인하세요.",
    ]
    if IS_WINDOWS:
        tips += [
            "설정 > 개인 정보 및 보안 > 마이크 에서 '데스크톱 앱이 마이크에 액세스하도록 허용'을 켜세요.",
            "설정 > 시스템 > 소리 > 입력 에서 장치가 '허용'이고 입력 볼륨이 0이 아닌지 확인하세요.",
        ]
    elif IS_LINUX:
        tips += [
            "arecord -l 로 마이크가 인식되는지, pavucontrol 에서 음소거가 아닌지 확인하세요.",
            "'hw:' 장치가 안 열리면 --mic pulse 또는 --mic default 를 쓰세요.",
        ]
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
        self.pos = 0.0  # 다음 출력 샘플의 위치 (self.buf 기준, 소수)
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
                self.sd.check_input_settings(
                    device=self.device, samplerate=rate, channels=channels, dtype="int16"
                )
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

        with self.sd.RawInputStream(
            samplerate=self.rate,
            blocksize=blocksize,
            device=self.device,
            channels=self.channels,
            dtype="int16",
        ) as stream:
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


def open_microphone(device: int | str | None) -> Microphone:
    sd = import_sounddevice()
    try:
        mic = Microphone(sd, device)
    except MicrophoneError as e:
        target = "기본 입력 장치(마이크)" if device is None else f"입력 장치 {device!r}"
        raise SttError(f"[오류] {target}를 열 수 없습니다: {e}\n{mic_help()}") from e
    conversion = f" ({mic.rate}Hz {mic.channels}ch로 열고 16kHz mono로 변환)" if mic.needs_conversion else ""
    log(f"[마이크] {mic.name}{conversion}")
    return mic


def load_wav(path: str | Path) -> np.ndarray:
    """wav -> 16kHz mono float32 [-1, 1]. ffmpeg 없이 scipy 로 읽는다."""
    from math import gcd

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128) / 128
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return x


def save_wav(path: Path, pcm: bytes) -> None:
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, SAMPLE_RATE, np.frombuffer(pcm, dtype=np.int16))


# --------------------------------------------------------------------------- #
# 녹음: Enter 모드
# --------------------------------------------------------------------------- #
def line_ready(stream, timeout: float = 0.0) -> bool:
    """stream 에서 한 줄을 막힘 없이 읽을 수 있는지. 터미널(canonical)은 Enter 후에 True."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):  # StringIO 등 (테스트) -> 항상 읽기 가능
        return True
    if IS_WINDOWS:
        import msvcrt

        end = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.01)
    import select

    return bool(select.select([fd], [], [], timeout)[0])


def record_until_enter(
    frames: Iterable[bytes],
    stop_requested: Callable[[], bool],
    frame_ms: int,
    max_record_s: float,
    post_roll_ms: int = POST_ROLL_MS,
) -> tuple[bytes, bool]:
    """stop_requested() 가 True 가 될 때(Enter)까지 녹음. (pcm, Enter로_끝났는지) 반환.

    Enter 뒤에도 post_roll_ms 만큼 더 받는다. max_record_s 에 닿으면 강제로 끊는다.
    """
    max_frames = max(1, int(max_record_s * 1000 // frame_ms))
    post = post_roll_ms // frame_ms
    recorded: list[bytes] = []
    stopped_at = None
    for frame in frames:
        recorded.append(frame)
        if stopped_at is None and stop_requested():
            stopped_at = len(recorded)
        if stopped_at is not None and len(recorded) >= stopped_at + post:
            break
        if len(recorded) >= max_frames:
            break
    return b"".join(recorded), stopped_at is not None


# --------------------------------------------------------------------------- #
# 녹음: VAD 모드 (원본 그대로)
# --------------------------------------------------------------------------- #
class StreamWindows:
    """마이크를 별도 스레드가 계속 읽고, hop 마다 최근 window 오디오를 (최근 1초에 말소리가 있으면) 내준다.

    '말이 끝날 때까지 녹음' 은 팬·서보 소음이 계속되면 말 끝을 못 찾아 최대 녹음 길이를 다 채운 뒤에야
    반응한다 (실측: 대기 명령 10s, 붓는 중 정지 4.3s). 겹치는 창은 소음과 무관하게 ≈ hop + 변환 시간에 반응하고,
    명령어가 창 경계에서 잘려도 다음 창에 온전히 들어간다. 읽기를 스레드로 나눈 이유: 변환(≈0.3s) 동안
    읽기를 멈추면 입력 버퍼가 넘쳐 소리가 유실된다 ("정지했습니다" → "제했습니다").

        with StreamWindows(mic, vad) as sw:
            pcm = sw.poll()      # bytes (16 kHz int16) 또는 None, 기다리지 않음
    """

    def __init__(self, mic, vad, frame_ms=30, window_s=2.0, hop_s=0.5, recent_s=1.0, min_speech=0.3):
        import threading
        from collections import deque

        self.mic, self.vad, self.frame_ms = mic, vad, frame_ms
        self.n_win = max(1, int(window_s * 1000 / frame_ms))
        self.n_hop = max(1, int(hop_s * 1000 / frame_ms))
        self.n_recent = max(1, min(self.n_win, int(recent_s * 1000 / frame_ms)))
        self.min_speech = min_speech
        self.ring: deque = deque(maxlen=self.n_win)
        self.lock, self.stop_ev = threading.Lock(), threading.Event()
        self.count, self.last, self.error = 0, 0, None
        self._thread = threading.Thread(target=self._read, daemon=True)

    def _read(self):
        try:
            for f in self.frames:
                sp = self.vad.is_speech(f, SAMPLE_RATE)
                with self.lock:
                    self.ring.append((f, sp))
                    self.count += 1
                if self.stop_ev.is_set():
                    return
        except Exception as e:  # 마이크 끊김 등 → poll 쪽에서 다시 던진다
            self.error = e

    def __enter__(self):
        self.frames = self.mic.frames(self.frame_ms)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_ev.set()
        self._thread.join(timeout=1.0)
        self.frames.close()

    def clear(self):
        with self.lock:
            self.ring.clear()

    def poll(self) -> bytes | None:
        if self.error is not None:
            raise self.error
        with self.lock:
            # 창이 다 차기 전(마이크를 막 연 직후)에도 최근 1s 만 모이면 본다. 2s 를 기다리면 안내 음성 직후
            # 바로 한 말("정지했습니다." → "계속 해줘")이 창이 찰 때쯤엔 '최근 1s' 밖으로 밀려나 버려졌다.
            if self.count - self.last < self.n_hop or len(self.ring) < self.n_recent:
                return None
            self.last = self.count
            win = list(self.ring)
        if sum(sp for _, sp in win[-self.n_recent :]) < self.min_speech * self.n_recent:
            return None
        return b"".join(f for f, _ in win)


def import_webrtcvad():
    try:
        import warnings

        with warnings.catch_warnings():  # pkg_resources deprecation 경고 숨김
            warnings.simplefilter("ignore")
            import webrtcvad
    except ImportError as e:
        if "pkg_resources" in str(e):  # 최신 setuptools에서 pkg_resources가 빠진 경우
            raise SttError(
                f"[오류] webrtcvad import 실패: {e}\n"
                '       pip install "setuptools<81"  또는  pip install webrtcvad-wheels'
            ) from e
        if IS_WINDOWS:
            raise SttError(
                "[오류] webrtcvad가 없습니다: pip install webrtcvad-wheels\n"
                "       (원본 webrtcvad 패키지는 Visual C++ Build Tools가 있어야 설치됩니다)"
            ) from e
        raise SttError(
            "[오류] webrtcvad가 없습니다: pip install webrtcvad\n"
            "       (빌드 실패 시 sudo apt install build-essential python3-dev 후 재시도,\n"
            "        또는 미리 빌드된 pip install webrtcvad-wheels)"
        ) from e
    return webrtcvad


def record_utterance(
    frames: Iterable[bytes],
    vad,
    frame_ms: int,
    silence_ms: int,
    timeout_s: float,
    max_record_s: float,
    continue_vad=None,
) -> bytes | None:
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
            log(
                f"[경고] 최대 녹음 길이 {max_record_s:g}초 도달 - 강제 종료 "
                "(주변 소음이 크면 --vad-aggressiveness를 높여보세요)"
            )
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
def prefer_passive_omp() -> None:
    """OpenMP 대기 스레드가 바쁜 대기(spin) 대신 잠들게 한다 (torch import 전에만 효과, 이미 지정돼 있으면 그대로).

    Whisper small CPU 측정: 같은 지연에서 CPU 사용량 20 -> 14 코어·초/명령 (spin 낭비 제거), 다른 작업과 덜 다툰다.
    CLI 진입점에서만 부른다. 라이브러리로 쓸 때는 호출 쪽 프로세스(정책 추론 등) 전체에 영향을 주므로 정하지 않는다.
    """
    if "torch" not in sys.modules:
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def load_whisper(model_name: str, device: str):
    try:
        import torch
        import whisper  # 무거운 import는 실제로 필요할 때만
    except ImportError as e:
        raise SttError("[오류] openai-whisper가 없습니다: pip install openai-whisper") from e

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise SttError(
            "[오류] device=cuda 를 지정했지만 CUDA를 사용할 수 없습니다. --stt-device cpu 로 실행하세요."
        )

    log(f"[STT] Whisper '{model_name}' 모델 로딩 중 ({device}) - 첫 실행이면 다운로드에 시간이 걸립니다...")
    try:
        model = whisper.load_model(model_name, device=device)
    except RuntimeError as e:  # 잘못된 모델 이름, GPU 메모리 부족 등
        raise SttError(
            f"[오류] Whisper 모델 로딩 실패: {e}\n"
            "       GPU 메모리 부족이면 --stt-device cpu 또는 더 작은 --model 을 쓰세요."
        ) from e
    return model, device


def default_fp16(device: str) -> bool:
    """fp16 은 텐서 코어가 있는 GPU(compute capability 7.0+)에서만 빠르다.

    Pascal(GTX 10xx, 6.x)은 fp16 처리량이 fp32 의 1/64 라 오히려 느리다
    (GTX 1060 + small: fp16 1.8s, fp32 0.3s, CPU 1.2s). CPU 는 fp16 미지원.
    """
    if device != "cuda":
        return False
    import torch

    return torch.cuda.get_device_capability()[0] >= 7


def transcribe(
    model,
    audio: np.ndarray,
    fp16: bool,
    initial_prompt: str | None = INITIAL_PROMPT,
    temperature=0.0,
    sample_len: int | None = SAMPLE_LEN,
    without_timestamps: bool = True,
) -> str:
    """16kHz mono float32 -> 텍스트. 지연시간 상한을 두는 설정으로 부른다.

    temperature=0.0     : 한 번만 디코딩 (greedy, beam 없음). Whisper 기본값 (0.0, 0.2, ..., 1.0) 은 잡음에서
                          재시도를 거듭해 CPU 로 수십 초가 걸릴 수 있어서 짧은 명령어에는 쓰지 않는다.
    without_timestamps  : 타임스탬프 토큰을 만들지 않는다. 30초 창 하나를 정확히 한 번만 디코딩한다
                          (타임스탬프가 있으면 남은 구간을 다시 디코딩하는 경우가 생긴다).
    sample_len          : 생성 토큰 상한 (SAMPLE_LEN 참고). None = Whisper 기본(224).
    """
    opts = {"without_timestamps": without_timestamps}
    if sample_len:
        opts["sample_len"] = sample_len
    result = model.transcribe(
        audio.astype(np.float32),
        language="ko",
        task="transcribe",
        fp16=fp16,  # CPU는 fp16 미지원
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
        temperature=temperature,
        **opts,
    )
    return result["text"].strip()


class torch_threads:
    """with torch_threads(n): 그 안에서만 torch CPU 스레드 수를 n 으로 (끝나면 원래대로). n=None 이면 그대로.

    같은 프로세스의 정책(ACT) 추론 스레드 설정을 건드리지 않으려고 전역으로 바꾸지 않는다.
    """

    def __init__(self, n: int | None):
        self.n, self.old = n, None

    def __enter__(self):
        if self.n:
            import torch

            self.old = torch.get_num_threads()
            if self.old != self.n:
                torch.set_num_threads(self.n)
        return self

    def __exit__(self, *exc):
        if self.n and self.old and self.old != self.n:
            import torch

            torch.set_num_threads(self.old)
        return False


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# 라이브러리 API
# --------------------------------------------------------------------------- #
class VoiceCommander:
    """Whisper 를 한 번 로딩해 두고 명령을 반복해서 받는다.

    model: Whisper 크기 이름 또는 이미 로딩한 모델 객체. device: "auto" | "cpu" | "cuda".
    mic: 입력 장치 번호/이름 (None = 시스템 기본). 마이크는 listen() 처음 호출 때 연다
    (transcribe_file 만 쓸 때는 PortAudio 가 없어도 된다).
    sample_len: Whisper 생성 토큰 상한 (지연시간 상한). num_threads: 변환할 때만 쓸 torch CPU 스레드 수
    (None = torch 기본). warmup: 모델 이름으로 로딩했을 때 1초 잡음을 한 번 변환해 첫 명령의 지연을 없앤다.
    """

    def __init__(
        self,
        model="small",
        device: str = "auto",
        mic: int | str | None = None,
        max_record_s: float = 10.0,
        initial_prompt: str | None = INITIAL_PROMPT,
        fp16: bool | None = None,
        temperature=0.0,
        min_peak: int = MIN_PEAK,
        frame_ms: int = 30,
        stdin=None,
        save_wav_path: Path | None = None,
        sample_len: int | None = SAMPLE_LEN,
        num_threads: int | None = None,
        warmup: bool = True,
    ):
        if isinstance(model, str):
            t = time.perf_counter()
            self.model, self.device = load_whisper(model, device)
            self.load_s = time.perf_counter() - t
            log(f"[STT] 로딩 완료 ({self.load_s:.1f}s)")
        else:
            warmup = False  # 주어진 모델 객체(테스트의 가짜 모델 등)는 건드리지 않는다
            self.model, self.device, self.load_s = model, ("cpu" if device == "auto" else device), 0.0
        self.fp16 = default_fp16(self.device) if fp16 is None else fp16
        self.mic_device = mic
        self.max_record_s = max_record_s
        self.initial_prompt = initial_prompt
        self.temperature = temperature
        self.min_peak = min_peak
        self.frame_ms = frame_ms
        self.stdin = stdin if stdin is not None else sys.stdin
        self.save_wav_path = save_wav_path
        self.sample_len = sample_len
        self.num_threads = num_threads
        self._mic: Microphone | None = None
        if warmup:
            self.warmup()

    def warmup(self) -> float:
        """잡음 1초를 한 번 변환한다 (첫 호출에만 생기는 초기화 지연을 미리 치름). 걸린 시간(초) 반환."""
        t = time.perf_counter()
        noise = np.random.default_rng(0).normal(0, 0.05, SAMPLE_RATE).astype(np.float32)
        with torch_threads(self.num_threads):
            transcribe(self.model, noise, self.fp16, self.initial_prompt, self.temperature, self.sample_len)
        return time.perf_counter() - t

    @property
    def mic(self) -> Microphone:
        if self._mic is None:
            self._mic = open_microphone(self.mic_device)
        return self._mic

    # ---- 변환 ----
    def transcribe_audio(self, audio: np.ndarray) -> CommandResult:
        """16kHz mono float32 배열 -> CommandResult(source="voice")."""
        duration = len(audio) / SAMPLE_RATE
        if len(audio) == 0 or np.abs(audio).max() * 32768 < self.min_peak:
            return CommandResult("", None, "voice", duration, 0.0)  # 무음: Whisper 환각 방지
        t = time.perf_counter()
        with torch_threads(self.num_threads):
            text = transcribe(
                self.model, audio, self.fp16, self.initial_prompt, self.temperature, self.sample_len
            )
        stt_s = time.perf_counter() - t
        return CommandResult(text, match_command(text), "voice", duration, stt_s)

    def transcribe_pcm(self, pcm: bytes) -> CommandResult:
        return self.transcribe_audio(pcm_to_float(pcm))

    def transcribe_file(self, path: str | Path) -> CommandResult:
        return self.transcribe_audio(load_wav(path))

    # ---- 키보드 ----
    def _readline(self) -> str:
        return self.stdin.readline()

    def poll_keyboard(self) -> str | None:
        """non-blocking. 한 줄이 입력돼 있으면 해석해서 명령("pour"/"stop"/"quit")을 반환.

        로봇 동작 중 s+Enter 로 멈추는 용도. 입력이 없거나 명령이 아니면 None.
        """
        if not line_ready(self.stdin, 0.0):
            return None
        line = self._readline()
        if line == "":
            return None
        _, cmd = parse_enter_input(line)
        return cmd

    # ---- Enter 모드 ----
    def listen(self) -> CommandResult:
        """Enter 로 녹음 시작/종료 후 인식. 키를 치면 그 키의 명령을 바로 반환.

        command 가 None 이면 인식 실패 (호출 쪽에서 다시 listen() 하면 된다).
        stdin 이 닫히면(EOF) command="quit".
        """
        while True:
            log(f"\n[대기] Enter = 녹음 시작  |  {KEY_HINT}")
            line = self._readline()
            if line == "":
                return CommandResult("", QUIT, "keyboard")
            action, cmd = parse_enter_input(line)
            if action in ("command", "quit"):
                log(f"[키보드] {line.strip()!r} -> {cmd}")
                return CommandResult(line.strip(), cmd, "keyboard")
            if action == "unknown":
                log(f"[안내] 알 수 없는 입력 {line.strip()!r}. {KEY_HINT}")
                continue
            break

        log(f"[녹음] 말씀하세요... 끝나면 Enter (최대 {self.max_record_s:g}초)")
        stop_line: list[str] = []

        def stop_requested() -> bool:
            if line_ready(self.stdin, 0.0):
                stop_line.append(self._readline())
                return True
            return False

        mic = self.mic
        with closing(mic.frames(self.frame_ms)) as frames:
            pcm, by_enter = record_until_enter(frames, stop_requested, self.frame_ms, self.max_record_s)
        if not by_enter:
            log(f"[녹음] 최대 길이 {self.max_record_s:g}초 도달 - 자동 종료")
        # 녹음 중에 p/s/q 를 쳐서 끝냈으면 키보드 명령 우선
        if stop_line:
            action, cmd = parse_enter_input(stop_line[0])
            if action in ("command", "quit"):
                log(f"[키보드] {stop_line[0].strip()!r} -> {cmd}")
                return CommandResult(stop_line[0].strip(), cmd, "keyboard")
        return self._finish(pcm, mic)

    # ---- VAD 모드 ----
    def listen_vad(
        self, vad, continue_vad, silence_ms: int = 1200, timeout_s: float = 15.0
    ) -> CommandResult | None:
        """말소리 자동 감지 녹음 후 인식. 타임아웃이면 None."""
        log(f"\n[대기] 말씀하세요... ({timeout_s:g}초 동안 말이 없으면 타임아웃)")
        mic = self.mic
        with closing(mic.frames(self.frame_ms)) as frames:
            pcm = record_utterance(
                frames, vad, self.frame_ms, silence_ms, timeout_s, self.max_record_s, continue_vad
            )
        if pcm is None:
            log(f"[타임아웃] {timeout_s:g}초 동안 말소리가 감지되지 않았습니다.")
            if mic.peak == 0:
                log(
                    "[안내] 마이크 입력이 완전히 0(무음)입니다. 음소거되었거나 마이크 권한이 막혀 있을 수 있습니다.\n"
                    + mic_help()
                )
            return None
        return self._finish(pcm, mic)

    def listen_auto(
        self, vad_level: int = 2, silence_ms: int = 1200, vad=None, continue_vad=None, chunk_s: float = 30.0
    ) -> CommandResult:
        """Enter 없이: 말소리가 들리면 자동 녹음 → 인식. 말이 없으면 조용히 계속 기다린다.

        기다리는 동안에도 p/s/q + Enter 키보드 입력을 받는다 (키가 우선).
        마이크 스트림은 호출마다 새로 연다 → 호출 전에 재생된 안내 음성(TTS)은 들어가지 않는다.
        호출 쪽은 TTS 재생이 끝난 뒤에 부를 것 ("물을 따르겠습니다" 에 '따르' 가 있어 스스로 명령이 될 수 있음).
        """
        if vad is None:
            if getattr(self, "_vads", None) is None:
                webrtcvad = import_webrtcvad()
                self._vads = (webrtcvad.Vad(vad_level), webrtcvad.Vad(max(0, vad_level - 1)))
            vad, continue_vad = self._vads
        key: list[str] = []
        kb_alive = [True]

        def frames_with_keys(frames):
            for f in frames:
                if kb_alive[0] and line_ready(self.stdin, 0.0):
                    line = self._readline()
                    if line == "":  # EOF (파이프 등) → 키보드 감시 끔
                        kb_alive[0] = False
                    else:
                        action, cmd = parse_enter_input(line)
                        if action in ("command", "quit"):
                            key.append(line.strip())
                            return
                yield f

        log(f'\n[대기] 말씀하세요 ("물 따라줘")  |  {KEY_HINT}')
        mic = self.mic
        while True:
            with closing(mic.frames(self.frame_ms)) as frames:
                pcm = record_utterance(
                    frames_with_keys(frames),
                    vad,
                    self.frame_ms,
                    silence_ms,
                    chunk_s,
                    self.max_record_s,
                    continue_vad,
                )
            if key:
                _, cmd = parse_enter_input(key[0])
                log(f"[키보드] {key[0]!r} -> {cmd}")
                return CommandResult(key[0], cmd, "keyboard")
            if pcm is not None:
                return self._finish(pcm, mic)
            if mic.peak == 0:  # chunk_s 동안 완전 무음 → 마이크 문제일 가능성만 알린다
                log("[안내] 마이크 입력이 완전히 0(무음)입니다.\n" + mic_help())

    def listen_stream(
        self, vad_level: int = 3, window_s: float = 2.0, hop_s: float = 0.5, pour_hits: int = 2
    ) -> CommandResult:
        """실시간에 가깝게: 최근 window_s 를 hop_s 마다 받아써 명령이 나오면 바로 반환 (StreamWindows).

        말이 끝나기를 기다리지 않아 소음 속에서도 반응이 ≈1s. 키보드 p/s/q + Enter 도 받는다 (키가 우선).
        안전: 정지는 1번에 반환, 시작(pour)은 겹치는 창 연속 pour_hits 번 나와야 반환 (환각 한 번에 로봇이 움직이지 않게).
        Whisper 힌트 문장은 끈다 (잡음 → 힌트 문장 환각이 명령이 되므로).
        """
        if getattr(self, "_stream_vad", None) is None:
            self._stream_vad = import_webrtcvad().Vad(vad_level)
        log(f'\n[대기] 말씀하세요 ("물 따라줘")  |  {KEY_HINT}')
        kb_alive, hits, last_cmd = True, 0, None
        with StreamWindows(self.mic, self._stream_vad, self.frame_ms, window_s, hop_s) as sw:
            while True:
                if kb_alive and line_ready(self.stdin, 0.02):
                    line = self._readline()
                    if line == "":  # EOF (파이프 등) → 키보드 감시 끔
                        kb_alive = False
                    else:
                        action, cmd = parse_enter_input(line)
                        if action in ("command", "quit"):
                            log(f"[키보드] {line.strip()!r} -> {cmd}")
                            return CommandResult(line.strip(), cmd, "keyboard")
                elif not kb_alive:
                    time.sleep(0.02)
                pcm = sw.poll()
                if pcm is None:
                    continue
                audio = pcm_to_float(pcm)
                if np.abs(audio).max() * 32768 < self.min_peak:
                    continue
                t = time.perf_counter()
                with torch_threads(self.num_threads):
                    text = transcribe(self.model, audio, self.fp16, None, self.temperature, self.sample_len)
                stt_s = time.perf_counter() - t
                cmd = match_command(text)
                if text:
                    log(f'[듣는 중] "{text}" -> {cmd} ({stt_s:.2f}s)')
                # 로봇을 움직이는 명령(시작·초기자세)은 같은 명령이 연속 pour_hits 번 들려야 반환
                hits = (
                    hits + 1 if cmd in (POUR, HOME) and cmd == last_cmd else (1 if cmd in (POUR, HOME) else 0)
                )
                last_cmd = cmd
                if cmd == STOP or (cmd in (POUR, HOME) and hits >= pour_hits):
                    log(f"[명령] {cmd}")
                    return CommandResult(text, cmd, "voice", window_s, stt_s)

    def _finish(self, pcm: bytes, mic: Microphone) -> CommandResult:
        log(f"[녹음] 종료 ({len(pcm) / 2 / SAMPLE_RATE:.1f}초) - 텍스트 변환 중...")
        if self.save_wav_path:
            save_wav(self.save_wav_path, pcm)
        r = self.transcribe_pcm(pcm)
        if not r.text:
            log("[안내] 말소리가 들리지 않았습니다 (입력이 너무 작음).")
            if mic.peak == 0:
                log("[안내] 마이크 입력이 완전히 0(무음)입니다.\n" + mic_help())
        else:
            log(f'[인식] "{r.text}"  ({r.stt_s:.1f}s)')
        if r.command is None:
            log(f'[안내] 명령을 찾지 못했습니다. 예: "물 따라줘", "멈춰". {KEY_HINT}')
        else:
            log(f"[명령] {r.command}")
        return r


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="음성 명령 인식: 음성 -> 텍스트(Whisper, ko) -> 명령(pour/stop) -> command.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        default="enter",
        choices=("enter", "vad", "stream"),
        help="enter: Enter로 녹음 시작/종료, vad: 말소리 감지 후 말 끝까지 녹음, "
        "stream: 최근 2초를 0.5초마다 받아써 바로 반응 (소음에 강함)",
    )
    p.add_argument("--wav", type=Path, nargs="+", default=None, help="마이크 대신 wav 파일들을 인식")
    p.add_argument("--model", default="small", help="Whisper 모델 크기 (tiny/base/small/medium/large/turbo)")
    p.add_argument("--stt-device", default="auto", choices=("auto", "cpu", "cuda"), help="Whisper 실행 장치")
    p.add_argument("--initial-prompt", default=INITIAL_PROMPT, help="Whisper 어휘 힌트 ('' = 사용 안 함)")
    p.add_argument(
        "--sample-len", type=int, default=SAMPLE_LEN, help="Whisper 생성 토큰 상한 (0 = Whisper 기본 224)"
    )
    p.add_argument(
        "--threads", type=int, default=None, help="Whisper 변환 때 torch CPU 스레드 수 (기본: torch 기본)"
    )
    p.add_argument("--once", action="store_true", help="한 번만 인식하고 종료 (기본: 반복)")
    p.add_argument("--max-record-s", type=float, default=10.0, help="최대 녹음 길이(초)")
    p.add_argument("--silence-ms", type=int, default=1200, help="[vad] 이 시간(ms) 무음이면 녹음 종료")
    p.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=range(4),
        help="[vad] 민감도 0(관대)~3(엄격). 시끄러우면 높이세요",
    )
    p.add_argument("--timeout", type=float, default=15.0, help="[vad] 말소리가 없을 때 대기 시간(초)")
    p.add_argument("--frame-ms", type=int, default=30, choices=(10, 20, 30), help="오디오 프레임 길이(ms)")
    p.add_argument(
        "--mic", type=parse_device, default=None, help="입력 장치 번호 또는 이름 일부 (기본: 시스템 기본)"
    )
    p.add_argument("--list-devices", action="store_true", help="오디오 장치 목록 출력 후 종료")
    p.add_argument("--output", type=Path, default=OUT_DIR / "command.json", help="결과 JSON 경로")
    p.add_argument(
        "--save-wav", type=Path, default=None, help="마지막 녹음을 wav로 저장 (디버깅/테스트 클립 수집)"
    )
    args = p.parse_args(argv)
    if args.silence_ms <= 0 or args.timeout <= 0 or args.max_record_s <= 0:
        p.error("--silence-ms, --timeout, --max-record-s 는 0보다 커야 합니다")
    return args


def exit_code(r: CommandResult | None) -> int:
    if r is None:
        return EXIT_TIMEOUT
    return EXIT_OK if r.command else EXIT_NO_COMMAND


def run_wav(args, vc: VoiceCommander) -> int:
    code = EXIT_OK
    for path in args.wav:
        r = vc.transcribe_file(path)
        log(f'[{path.name}] "{r.text}" -> {r.command}  ({r.duration_s:.1f}s 음성, STT {r.stt_s:.2f}s)')
        save_command(args.output, r)
        code = max(code, exit_code(r))
    log(f"[저장] {args.output}")
    return code


def main(argv: list[str] | None = None) -> int:
    configure_console()
    prefer_passive_omp()
    args = parse_args(argv)
    try:
        if args.list_devices:
            print(import_sounddevice().query_devices())
            return EXIT_OK
        vad = continue_vad = mic = None
        if args.wav is None:
            if args.mode == "vad":
                webrtcvad = import_webrtcvad()
                vad = webrtcvad.Vad(args.vad_aggressiveness)  # 녹음 시작: 엄격히
                continue_vad = webrtcvad.Vad(max(0, args.vad_aggressiveness - 1))  # 녹음 유지: 관대히
            mic = open_microphone(args.mic)  # 모델 로딩 전에 마이크 문제부터 빨리 알려준다
        vc = VoiceCommander(
            args.model,
            args.stt_device,
            args.mic,
            args.max_record_s,
            args.initial_prompt,
            frame_ms=args.frame_ms,
            save_wav_path=args.save_wav,
            sample_len=args.sample_len or None,
            num_threads=args.threads,
        )
        if args.wav is not None:
            return run_wav(args, vc)
        vc._mic = mic
        while True:
            r = (
                vc.listen()
                if args.mode == "enter"
                else vc.listen_stream()
                if args.mode == "stream"
                else vc.listen_vad(vad, continue_vad, args.silence_ms, args.timeout)
            )
            if r is not None and r.command == QUIT:
                log("[종료] q")
                return EXIT_OK
            if r is not None:
                save_command(args.output, r)
                log(f"[저장] {args.output}  {json.dumps(asdict(r), ensure_ascii=False)}")
            if args.once:
                return exit_code(r)
    except SttError as e:
        log(str(e))
        return EXIT_ERROR
    except KeyboardInterrupt:
        log("\n[종료] Ctrl+C")
        return EXIT_OK
    except OSError as e:  # JSON 저장 실패 등
        log(f"[오류] {e}")
        return EXIT_ERROR
    except Exception as e:  # sounddevice.PortAudioError (마이크 연결 해제 등)
        if type(e).__name__ != "PortAudioError":
            raise
        log(f"[오류] 오디오 입력 중 문제가 발생했습니다 (마이크 연결 해제?): {e}\n{mic_help()}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
