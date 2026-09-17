"""마이크/Whisper 없이 돌릴 수 있는 로직 테스트.

    pytest tests/                               # pytest가 있으면
    python tests/test_voice_command_vad.py      # 없어도 실행 가능
"""

import json
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice_command_vad as v  # noqa: E402

S, N = 1, 0  # 가짜 프레임: 1 = 음성, 0 = 무음


class FakeVad:
    def is_speech(self, frame, sample_rate):
        return frame[0] == S


def run_vad(pattern, **kwargs):
    opts = dict(frame_ms=30, silence_ms=800, timeout_s=15, max_record_s=15)
    opts.update(kwargs)
    return v.record_utterance((bytes([f]) for f in pattern), FakeVad(), **opts)


# --------------------------------------------------------------------------- #
# 색상 추출 / JSON
# --------------------------------------------------------------------------- #
def test_extract_color():
    cases = {
        "빨간색 컵 집어줘": "red", "레드": "red", "적색 물체": "red", "빨 간색 컵": "red",
        "파랑 블록 옮겨": "blue", "청색": "blue",
        "초록색": "green", "녹색 컵": "green",
        "노란 거": "yellow", "옐로우 공": "yellow",
        "빨간 컵을 파란 상자에 넣어줘": "red", "파란 상자에 빨간 컵": "blue",  # 먼저 나온 색
        "컵 좀 집어줘": None, "": None,
    }
    for text, expected in cases.items():
        assert v.extract_color(text) == expected, text


def test_save_command():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "command.json"
        v.save_command(path, "컵 집어줘", None)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["text"] == "컵 집어줘" and data["color"] is None and "T" in data["timestamp"]
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
            v.save_command(path, "빨간 컵", "red")
            assert json.loads(path.read_text(encoding="utf-8"))["color"] == "red"
            assert calls["n"] == 4
    finally:
        v.os.replace = real_replace


# --------------------------------------------------------------------------- #
# VAD 상태 머신
# --------------------------------------------------------------------------- #
def test_vad_records_utterance_with_preroll_and_trimmed_tail():
    out = run_vad([N] * 33 + [S] * 50 + [N] * 67)
    assert out.count(S) == 50
    assert 0 < out.index(S) <= v.PRE_ROLL_MS // 30          # 시작 전 오디오 일부 포함
    keep = v.TRAILING_SILENCE_KEEP_MS // 30
    assert out.endswith(bytes([N] * keep)) and out[-keep - 1] == S


def test_vad_short_pause_stays_in_one_utterance():
    out = run_vad([N] * 10 + [S] * 20 + [N] * 20 + [S] * 20 + [N] * 40)  # 600ms 쉼 < 800ms
    assert out.count(S) == 40


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
# 리샘플링 / 마이크
# --------------------------------------------------------------------------- #
def test_stream_resampler_matches_ideal_signal():
    for rate, block in [(48000, 1440), (44100, 1323), (32000, 960), (22050, 661)]:
        t = np.arange(rate * 2) / rate
        x = 10000 * np.sin(2 * np.pi * 440 * t)
        r = v.StreamResampler(rate)
        out = np.concatenate([r.process(x[i:i + block]) for i in range(0, len(x), block)])
        assert abs(len(out) - 32000) <= 2, rate
        delay = (len(r.taps) - 1) / 2 / rate
        ref = 10000 * np.sin(2 * np.pi * 440 * (np.arange(len(out)) / 16000 - delay))
        sl = slice(1600, len(out) - 1600)
        snr = 10 * np.log10(np.sum(ref[sl] ** 2) / np.sum((out[sl] - ref[sl]) ** 2))
        assert snr > 30, (rate, snr)


def test_stream_resampler_suppresses_aliasing():
    x = 10000 * np.sin(2 * np.pi * 12000 * np.arange(48000) / 48000)  # 16kHz에서 표현 불가한 톤
    r = v.StreamResampler(48000)
    out = np.concatenate([r.process(x[i:i + 1440]) for i in range(0, 48000, 1440)])
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
    assert (a.model, a.once, a.silence_ms, a.vad_aggressiveness, a.timeout) == ("small", False, 800, 2, 15.0)
    a = v.parse_args(["--once", "--model", "base", "--mic", "3"])
    assert a.once and a.model == "base" and a.mic == 3
    for bad in (["--vad-aggressiveness", "4"], ["--frame-ms", "25"], ["--silence-ms", "0"]):
        try:
            v.parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


if __name__ == "__main__":
    import contextlib
    import io

    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} passed")
