"""마이크/Whisper 없이 돌릴 수 있는 로직 테스트 (lerobot_pac 원본 테스트를 옮기고 확장).

실행:
    cd stt && python -m pytest -q tests      # pytest가 있으면
    python stt/tests/test_voice_command.py   # 없어도 실행 가능
"""

import io
import json
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voice_command as v  # noqa: E402

S, N = 1, 0  # 가짜 프레임: 1 = 음성, 0 = 무음


class FakeVad:
    def is_speech(self, frame, sample_rate):
        return frame[0] == S


def run_vad(pattern, **kwargs):
    opts = dict(frame_ms=30, silence_ms=800, timeout_s=15, max_record_s=15)
    opts.update(kwargs)
    return v.record_utterance((bytes([f]) for f in pattern), FakeVad(), **opts)


# --------------------------------------------------------------------------- #
# 명령어 매칭
# --------------------------------------------------------------------------- #
POUR_CASES = [
    "물 따라줘",
    "물 좀 따라 줄래",
    "물 좀 따라 줄래?",
    "물 부어줘",
    "물 한 잔 줘",
    "물 좀 주세요",
    "물 따라 주세요",
    "따라줘",
    "부어 줘",
    "컵에 물 좀 부어 줄래?",
    "목마른데 물 좀 줄래?",
    "물좀따라줘",
    "물 따 라 줘",
    "물 한잔만 주세요.",
    "물 마시고 싶어",
    "물을 따라주세요!",
    "시원한 물 한 잔",
    "물 다라줘",  # 따->다 오인식도 "물+줘" 조합으로 잡힌다
]
STOP_CASES = [
    "멈춰",
    "멈춰!",
    "그만",
    "그만해",
    "정지",
    "스톱",
    "스탑",
    "Stop.",
    "STOP",
    "잠깐만",
    "멈 춰",
    "그 만",
    "중지해 줘",
]
STOP_PRIORITY_CASES = [  # pour 와 stop 이 같이 있으면 stop
    "그만 따라",
    "물 그만 따라줘",
    "따르지 마",
    "물 붓지 마",
    "물 따라줘 아니 멈춰",
    "멈춰 물 따라줘",
    "물 따라줘. 정지.",
    "물 따라줘. 멈춰. 그만.",  # 프롬프트 그대로 환각해도 stop
]
NONE_CASES = [
    "",
    "오늘 날씨 좋네요",
    "안녕하세요 반갑습니다",
    "이거 뭐예요?",
    "나를 따라와",
    "따라서 결론은",
    "선물 좀 줘",
    "물건 좀 집어줘",
    "동물 좋아해요",
    "물어볼 게 있어요",
    "시청해 주셔서 감사합니다",
    "법에 따르면",
    "따라가 보자",
]


def test_match_pour():
    for text in POUR_CASES:
        assert v.match_command(text) == "pour", text


def test_match_stop():
    for text in STOP_CASES:
        assert v.match_command(text) == "stop", text


def test_stop_wins_over_pour():
    for text in STOP_PRIORITY_CASES:
        assert v.match_command(text) == "stop", text


def test_match_none():
    for text in NONE_CASES:
        assert v.match_command(text) is None, text


def test_normalize():
    assert v.normalize(" 물 좀,  따라 줄래?! ") == "물좀따라줄래"
    assert v.normalize("Stop.") == "stop"


def test_custom_commands_and_priority():
    cmds = {"pour": ["따라"], "stop": ["멈춰"], "home": ["처음"]}
    assert v.match_command("처음 자세로", cmds, {}, ("stop", "pour", "home")) == "home"
    assert v.match_command("처음 멈춰", cmds, {}, ("stop", "pour", "home")) == "stop"
    # priority 에 없는 명령은 맨 뒤
    assert v.match_command("처음 따라", cmds, {}, ("stop", "pour")) == "pour"


# --------------------------------------------------------------------------- #
# Enter 모드 키 입력
# --------------------------------------------------------------------------- #
def test_parse_enter_input():
    cases = {
        "\n": ("record", None),
        "   \n": ("record", None),
        "p\n": ("command", "pour"),
        "P\n": ("command", "pour"),
        "ㅔ\n": ("command", "pour"),
        "s\n": ("command", "stop"),
        "ㄴ\n": ("command", "stop"),
        "q\n": ("quit", "quit"),
        "ㅂ\n": ("quit", "quit"),
        "quit\n": ("quit", "quit"),
        "물 따라줘\n": ("command", "pour"),
        "그만\n": ("command", "stop"),
        "x\n": ("unknown", None),
        "hello\n": ("unknown", None),
    }
    for line, expected in cases.items():
        assert v.parse_enter_input(line) == expected, line


def test_record_until_enter_post_roll():
    presses = iter([False] * 9 + [True])  # 10번째 프레임에서 Enter
    pcm, by_enter = v.record_until_enter(
        (bytes([S]) for _ in range(1000)), lambda: next(presses, True), frame_ms=30, max_record_s=10
    )
    assert by_enter and len(pcm) == 10 + v.POST_ROLL_MS // 30


def test_record_until_enter_max_length():
    pcm, by_enter = v.record_until_enter((bytes([S]) for _ in range(1000)), lambda: False, 30, 3)
    assert not by_enter and len(pcm) == 100


def test_record_until_enter_source_ends():
    pcm, by_enter = v.record_until_enter((bytes([S]) for _ in range(5)), lambda: False, 30, 10)
    assert not by_enter and len(pcm) == 5


# --------------------------------------------------------------------------- #
# VoiceCommander (가짜 Whisper / 마이크 / stdin)
# --------------------------------------------------------------------------- #
class FakeWhisper:
    def __init__(self, text="물 따라줘"):
        self.text, self.calls = text, []

    def transcribe(self, audio, **kw):
        self.calls.append((audio, kw))
        return {"text": " " + self.text + " "}


class FakeMic:
    peak = 10000

    def __init__(self, amp=8000):
        self.amp = amp

    def frames(self, frame_ms):
        n = v.SAMPLE_RATE * frame_ms // 1000
        while True:
            yield (np.full(n, self.amp, np.int16)).tobytes()


def make_vc(stdin_text, text="물 따라줘", amp=8000, **kw):
    vc = v.VoiceCommander(model=FakeWhisper(text), device="cpu", stdin=io.StringIO(stdin_text), **kw)
    vc._mic = FakeMic(amp)
    return vc


def test_listen_voice_and_whisper_options():
    vc = make_vc("\n\n")
    r = vc.listen()
    assert (r.command, r.source, r.text) == ("pour", "voice", "물 따라줘")
    assert r.duration_s > 0
    audio, kw = vc.model.calls[0]
    assert audio.dtype == np.float32
    assert kw["language"] == "ko" and kw["condition_on_previous_text"] is False
    assert kw["fp16"] is False and kw["initial_prompt"] == v.INITIAL_PROMPT and kw["temperature"] == 0.0


def test_listen_keyboard_fallback():
    for keys, cmd in [("p\n", "pour"), ("s\n", "stop"), ("q\n", "quit"), ("", "quit")]:  # "" = EOF
        vc = make_vc(keys)
        r = vc.listen()
        assert (r.command, r.source) == (cmd, "keyboard"), keys
        assert vc.model.calls == []  # Whisper 안 씀


def test_listen_unknown_key_then_record():
    vc = make_vc("xyz\n\n\n", text="그만")
    r = vc.listen()
    assert (r.command, r.source) == ("stop", "voice")


def test_listen_key_typed_during_recording_wins():
    vc = make_vc("\ns\n", text="물 따라줘")
    r = vc.listen()
    assert (r.command, r.source) == ("stop", "keyboard")


def test_listen_no_match_and_silence():
    r = make_vc("\n\n", text="오늘 날씨 좋네요").listen()
    assert r.command is None and r.text == "오늘 날씨 좋네요"
    vc = make_vc("\n\n", amp=10)  # 무음 -> Whisper 건너뜀
    r = vc.listen()
    assert r.command is None and r.text == "" and vc.model.calls == []


def test_initial_prompt_off_and_fp16_cuda(monkeypatch):
    for fast in (True, False):  # 텐서 코어 있는 GPU → fp16, Pascal 등 → fp32
        monkeypatch.setattr(v, "default_fp16", lambda device, fast=fast: fast)
        vc = v.VoiceCommander(model=FakeWhisper(), device="cuda", initial_prompt="")
        vc.transcribe_audio(np.full(16000, 0.3, np.float32))
        kw = vc.model.calls[0][1]
        assert kw["initial_prompt"] is None and kw["fp16"] is fast


def test_default_fp16_cpu_without_torch():
    assert v.default_fp16("cpu") is False


def test_transcribe_file_resamples():
    from scipy.io import wavfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pour__a.wav"
        t = np.arange(44100) / 44100
        st = (8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
        wavfile.write(path, 44100, np.stack([st, st], axis=1))
        vc = v.VoiceCommander(model=FakeWhisper("물 부어줘"), device="cpu")
        r = vc.transcribe_file(path)
        assert r.command == "pour" and abs(r.duration_s - 1.0) < 0.01
        audio = vc.model.calls[0][0]
        assert audio.ndim == 1 and abs(np.abs(audio).max() - 8000 / 32768) < 0.02


def test_poll_keyboard():
    assert make_vc("s\n").poll_keyboard() == "stop"
    assert make_vc("").poll_keyboard() is None
    assert make_vc("x\n").poll_keyboard() is None


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def test_save_command():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "command.json"
        v.save_command(path, v.CommandResult("물 따라줘", "pour", "voice", 1.2, 0.8))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"text", "command", "source", "timestamp"}
        assert (data["text"], data["command"], data["source"]) == ("물 따라줘", "pour", "voice")
        assert "T" in data["timestamp"]
        v.save_command(path, v.CommandResult("", None, "voice"))
        assert json.loads(path.read_text(encoding="utf-8"))["command"] is None
        assert not (path.parent / "command.json.tmp").exists()


def test_save_command_retries_when_file_locked():
    # Windows: 다른 프로세스가 파일을 열고 있으면 os.replace가 PermissionError
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("in use")
        return real_replace(src, dst)

    v.os.replace = flaky
    try:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "command.json"
            v.save_command(path, v.CommandResult("s", "stop", "keyboard"))
            assert json.loads(path.read_text(encoding="utf-8"))["command"] == "stop"
            assert calls["n"] == 4
    finally:
        v.os.replace = real_replace


# --------------------------------------------------------------------------- #
# VAD 상태 머신 (원본)
# --------------------------------------------------------------------------- #
def test_vad_records_utterance_with_preroll_and_trimmed_tail():
    out = run_vad([N] * 33 + [S] * 50 + [N] * 67)
    assert out.count(S) == 50
    assert 0 < out.index(S) <= v.PRE_ROLL_MS // 30  # 시작 전 오디오 일부 포함
    keep = v.TRAILING_SILENCE_KEEP_MS // 30
    assert out.endswith(bytes([N] * keep)) and out[-keep - 1] == S


def test_vad_short_pause_stays_in_one_utterance():
    out = run_vad([N] * 10 + [S] * 20 + [N] * 20 + [S] * 20 + [N] * 40)  # 600ms 쉼 < 800ms
    assert out.count(S) == 40


def test_vad_continue_vad_keeps_soft_speech():
    # 2 = 작은 말소리: 엄격한 VAD는 무음, 관대한 VAD는 음성으로 판정
    class Strict:
        def is_speech(self, frame, sample_rate):
            return frame[0] == S

    class Lenient:
        def is_speech(self, frame, sample_rate):
            return frame[0] in (S, 2)

    pattern = [N] * 10 + [S] * 20 + [2] * 40 + [S] * 20 + [N] * 60  # 작은 소리 1200ms 구간
    strict_only = v.record_utterance((bytes([f]) for f in pattern), Strict(), 30, 800, 15, 15)
    hysteresis = v.record_utterance((bytes([f]) for f in pattern), Strict(), 30, 800, 15, 15, Lenient())
    assert strict_only.count(S) == 20  # 작은 소리 구간에서 끊김
    assert hysteresis.count(S) == 40  # 끝까지 녹음


def test_vad_ignores_clicks_and_times_out():
    assert run_vad(([N] * 20 + [S]) * 30) is None


def test_vad_timeout_frame_count():
    consumed = 0

    def silence():
        nonlocal consumed
        while True:
            consumed += 1
            yield bytes([N])

    assert v.record_utterance(silence(), FakeVad(), 30, 800, 15, 15) is None
    assert consumed == 500  # 15초 / 30ms


def test_vad_max_record_length():
    assert len(run_vad([S] * 2000, max_record_s=3)) == 100


def test_vad_10ms_frames():
    out = run_vad([N] * 100 + [S] * 150 + [N] * 200, frame_ms=10)
    assert out.count(S) == 150 and out.endswith(bytes([N] * 30))


# --------------------------------------------------------------------------- #
# 리샘플링 / 마이크 (원본)
# --------------------------------------------------------------------------- #
def test_stream_resampler_matches_ideal_signal():
    for rate, block in [(48000, 1440), (44100, 1323), (32000, 960), (22050, 661)]:
        t = np.arange(rate * 2) / rate
        x = 10000 * np.sin(2 * np.pi * 440 * t)
        r = v.StreamResampler(rate)
        out = np.concatenate([r.process(x[i : i + block]) for i in range(0, len(x), block)])
        assert abs(len(out) - 32000) <= 2, rate
        delay = (len(r.taps) - 1) / 2 / rate
        ref = 10000 * np.sin(2 * np.pi * 440 * (np.arange(len(out)) / 16000 - delay))
        sl = slice(1600, len(out) - 1600)
        snr = 10 * np.log10(np.sum(ref[sl] ** 2) / np.sum((out[sl] - ref[sl]) ** 2))
        assert snr > 30, (rate, snr)


def test_stream_resampler_suppresses_aliasing():
    x = 10000 * np.sin(2 * np.pi * 12000 * np.arange(48000) / 48000)  # 16kHz에서 표현 불가한 톤
    r = v.StreamResampler(48000)
    out = np.concatenate([r.process(x[i : i + 1440]) for i in range(0, 48000, 1440)])
    assert 20 * np.log10(np.abs(out[800:-800]).max() / 10000) < -30


class FakePortAudioError(Exception):
    pass


class FakeWasapiSD:
    """WASAPI처럼 16kHz/mono는 거부하고 48kHz 스테레오만 여는 가짜 sounddevice."""

    PortAudioError = FakePortAudioError
    closed = False

    class RawInputStream:
        def __init__(self, samplerate, blocksize, device, channels, dtype):
            assert (samplerate, channels, dtype) == (48000, 2, "int16")
            self.t = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            FakeWasapiSD.closed = True

        def read(self, n):
            idx = np.arange(self.t, self.t + n)
            self.t += n
            s = (8000 * np.sin(2 * np.pi * 300 * idx / 48000)).astype(np.int16)
            return np.stack([s, s], axis=1).reshape(-1).tobytes(), False

    @staticmethod
    def query_devices(device, kind):
        return {"name": "Mic (USB)", "hostapi": 2, "default_samplerate": 48000.0, "max_input_channels": 2}

    @staticmethod
    def query_hostapis(index):
        return {"name": "Windows WASAPI"}

    @staticmethod
    def check_input_settings(device, samplerate, channels, dtype):
        if (samplerate, channels) != (48000, 2):
            raise FakePortAudioError("Invalid sample rate")


def test_microphone_falls_back_to_native_rate_and_converts():
    mic = v.Microphone(FakeWasapiSD, 5)
    assert (mic.rate, mic.channels, mic.needs_conversion) == (48000, 2, True)
    with closing(mic.frames(30)) as frames:
        got = [next(frames) for _ in range(40)]
    assert FakeWasapiSD.closed
    assert all(len(f) == 480 * 2 for f in got)  # 30ms @ 16kHz int16
    pcm = np.frombuffer(b"".join(got), np.int16).astype(float)[1600:]
    peak_hz = np.argmax(np.abs(np.fft.rfft(pcm))) * 16000 / len(pcm)
    assert abs(peak_hz - 300) < 5
    assert mic.peak >= 7999


def test_microphone_missing_device_raises():
    class NoMicSD(FakeWasapiSD):
        @staticmethod
        def query_devices(device, kind):
            raise FakePortAudioError("No input device available")

    try:
        v.Microphone(NoMicSD, None)
    except v.MicrophoneError:
        return
    raise AssertionError("MicrophoneError expected")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_parse_args():
    a = v.parse_args([])
    assert (a.mode, a.model, a.once, a.max_record_s, a.wav) == ("enter", "small", False, 10.0, None)
    assert a.initial_prompt == v.INITIAL_PROMPT and a.output == v.OUT_DIR / "command.json"
    a = v.parse_args(["--mode", "vad", "--once", "--model", "base", "--mic", "3"])
    assert a.mode == "vad" and a.once and a.model == "base" and a.mic == 3
    a = v.parse_args(["--wav", "a.wav", "b.wav", "--initial-prompt", ""])
    assert a.wav == [Path("a.wav"), Path("b.wav")] and a.initial_prompt == ""
    for bad in (
        ["--mode", "auto"],
        ["--vad-aggressiveness", "4"],
        ["--frame-ms", "25"],
        ["--silence-ms", "0"],
    ):
        try:
            v.parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_misheard_stop_and_story_exclusion():
    # Whisper 오인식 형태도 정지, "스토리" 같은 단어는 정지 아님
    assert v.match_command("정진.") == v.STOP
    assert v.match_command("스토") == v.STOP
    assert v.match_command("스톰") == v.STOP and v.match_command("스프") == v.STOP  # "스톱" 오인식
    assert v.match_command("스토리 들려줘") is None
    assert v.match_command("따라줘") == v.POUR


# --------------------------------------------------------------------------- #
# 독립성 / 출력 폴더 / 지연시간 상한 설정
# --------------------------------------------------------------------------- #
def test_no_imports_outside_stt_folder():
    # 폴더째 복사해 가도 동작하도록 형제 폴더(paths.py, pour_detector.py, tts/ 등)를 import 하지 않는다.
    import ast

    stt_dir = Path(v.__file__).resolve().parent
    local = {p.stem for p in stt_dir.glob("*.py")}
    third_party = {"numpy", "scipy", "torch", "whisper", "sounddevice", "webrtcvad", "edge_tts", "pytest"}
    allowed = set(sys.stdlib_module_names) | local | third_party
    for py in list(stt_dir.glob("*.py")) + list(stt_dir.glob("tests/*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, (py.name, name)


def test_default_out_dir(tmp_path=None):
    import contextlib

    with contextlib.ExitStack() as stack:
        if tmp_path is None:
            tmp_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        old = os.environ.pop("UNITA_LOCAL", None)
        from unittest import mock

        stack.enter_context(
            mock.patch.object(Path, "home", return_value=tmp_path / "home")
        )  # ~/UNITA_PAC2026/local 후보 차단
        try:
            fake = tmp_path / "proj" / "github" / "stt" / "voice_command.py"
            assert v.default_out_dir(fake) == fake.parent / "outputs"  # 폴더만 복사해 간 경우
            (tmp_path / "proj" / "local").mkdir(parents=True)
            assert v.default_out_dir(fake) == tmp_path / "proj" / "local" / "outputs" / "stt"
            os.environ["UNITA_LOCAL"] = str(tmp_path / "elsewhere")
            assert v.default_out_dir(fake) == tmp_path / "elsewhere" / "outputs" / "stt"
            assert v.default_out_dir(Path("/voice_command.py")) == tmp_path / "elsewhere" / "outputs" / "stt"
            del os.environ["UNITA_LOCAL"]
            assert v.default_out_dir(Path("/voice_command.py")) == Path("/outputs")  # 얕은 경로도 안 죽음
            (tmp_path / "home" / "UNITA_PAC2026" / "local").mkdir(parents=True)  # 홈 아래 프로젝트 local/
            other = tmp_path / "x" / "y" / "stt" / "voice_command.py"
            assert (
                v.default_out_dir(other) == tmp_path / "home" / "UNITA_PAC2026" / "local" / "outputs" / "stt"
            )
        finally:
            os.environ.pop("UNITA_LOCAL", None)
            if old is not None:
                os.environ["UNITA_LOCAL"] = old


def test_transcribe_latency_bounds():
    vc = v.VoiceCommander(model=FakeWhisper(), device="cpu")
    vc.transcribe_audio(np.full(16000, 0.3, np.float32))
    kw = vc.model.calls[0][1]
    assert kw["without_timestamps"] is True and kw["sample_len"] == v.SAMPLE_LEN == 48
    assert kw["temperature"] == 0.0 and "beam_size" not in kw  # greedy 한 번
    vc = v.VoiceCommander(model=FakeWhisper(), device="cpu", sample_len=None)
    vc.transcribe_audio(np.full(16000, 0.3, np.float32))
    assert "sample_len" not in vc.model.calls[0][1]


def test_warmup_only_when_loading_by_name():
    vc = v.VoiceCommander(model=FakeWhisper(), device="cpu")  # 모델 객체 -> 자동 워밍업 없음
    assert vc.model.calls == []
    assert vc.warmup() >= 0 and len(vc.model.calls) == 1


def test_torch_threads_restores():
    with v.torch_threads(None):  # None 이면 torch 를 import 하지도 않는다
        pass
    try:
        import torch
    except ImportError:
        return
    before = torch.get_num_threads()
    with v.torch_threads(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


def test_parse_args_latency_options():
    a = v.parse_args([])
    assert a.sample_len == v.SAMPLE_LEN and a.threads is None
    a = v.parse_args(["--sample-len", "0", "--threads", "4"])
    assert a.sample_len == 0 and a.threads == 4


if __name__ == "__main__":
    import contextlib

    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} passed")


# --------------------------------------------------------------------------- #
# Enter 없는 자동 듣기 (listen_auto)
# --------------------------------------------------------------------------- #
class PatternMic:
    """S/N 패턴대로 30ms 프레임(S = 진폭 8000, N = 0)을 주고, 다 쓰면 무음을 계속 준다."""

    peak = 10000

    def __init__(self, pattern):
        self.pattern = list(pattern)

    def frames(self, frame_ms):
        n = v.SAMPLE_RATE * frame_ms // 1000
        while True:
            f = self.pattern.pop(0) if self.pattern else N
            yield np.full(n, 8000 if f == S else 0, np.int16).tobytes()


class AmpVad:
    def is_speech(self, frame, sample_rate):
        return any(frame)


class LateStdin(io.StringIO):
    """after 번째 확인부터 한 줄이 준비된 것처럼 보이는 stdin (진짜 터미널처럼 그 전엔 읽을 게 없음)."""

    def __init__(self, text, after):
        super().__init__(text)
        self.after, self.checks = after, 0


def late_line_ready(stream, timeout=0.0):
    if isinstance(stream, LateStdin):
        stream.checks += 1
        return stream.checks >= stream.after
    return True


def make_auto(pattern, text="물 따라줘", stdin=None):
    vc = v.VoiceCommander(model=FakeWhisper(text), device="cpu", stdin=stdin or io.StringIO(""))
    vc._mic = PatternMic(pattern)
    return vc


def test_listen_auto_voice():
    vc = make_auto([N] * 20 + [S] * 30 + [N] * 50)
    vc._mic.peak = 10000
    r = vc.listen_auto(vad=AmpVad(), continue_vad=AmpVad(), silence_ms=600, chunk_s=5)
    assert (r.command, r.source) == ("pour", "voice")


def test_listen_auto_keeps_waiting_through_silence():
    # 첫 chunk(0.9s) 동안 무음 → 조용히 다시 기다렸다가 다음 chunk 에서 말소리
    vc = make_auto([N] * 40 + [S] * 30 + [N] * 50)
    r = vc.listen_auto(vad=AmpVad(), continue_vad=AmpVad(), silence_ms=600, chunk_s=0.9)
    assert r.command == "pour"


def test_listen_auto_keyboard_while_waiting(monkeypatch):
    monkeypatch.setattr(v, "line_ready", late_line_ready)
    vc = make_auto([N] * 1000, stdin=LateStdin("p\n", after=3))
    r = vc.listen_auto(vad=AmpVad(), continue_vad=AmpVad(), silence_ms=600, chunk_s=5)
    assert (r.command, r.source) == ("pour", "keyboard")
    vc = make_auto([N] * 1000, stdin=LateStdin("q\n", after=2))
    assert vc.listen_auto(vad=AmpVad(), continue_vad=AmpVad(), chunk_s=5).command == "quit"


def test_synonyms_idle():
    for text in ["컵 좀 채워줘", "물 담아줘", "목말라", "목이 말라요"]:
        assert v.match_command(text) == v.POUR, text
    for text in ["잠깐만요", "기다려", "멈출래", "어 저거 물 쏟아진다 멈춰"]:
        assert v.match_command(text) == v.STOP, text
    # 대회장 대화에서 잘못 걸린 말 (10/10 실측): 정지 동의어에서 빼거나 길어서 무시
    for text in [
        "다 됐어",
        "진짜 안 돼",
        "잠시만요",
        "여기 뭐예요? 여기 따라줘",
        "물 좀 채워주도록 하겠습니다",
        "이렇게 하지 못하는 사람을 멈춰 이렇게",
        "물 잡아줘",
        "술 따르기 때문에",
    ]:
        assert v.match_command(text) is None, text
    assert v.match_command("계속 따라줘") == v.POUR  # 대기 중엔 그대로 시작


def test_pause_back_resume():
    def m(t):
        return v.match_command(t, priority=v.PAUSE_PRIORITY)

    for text in [
        "돌아가",
        "제자리로",
        "원위치",
        "취소해줘",
        "되돌려줘",
        "내려놔",
        "안 할래",
        "필요 없어",
        "그만 돌아가",
    ]:
        assert m(text) == v.BACK, text
    for text in ["계속 해줘", "이어서 해줘", "마저 해줘", "진행해", "다시 해줘", "괜찮아"]:
        assert m(text) == v.RESUME, text
    assert m("마저 따라줘") == v.POUR  # 일시정지 중 pour = 계속
    assert m("멈춰") == v.STOP  # 이미 멈춤 → 무시
