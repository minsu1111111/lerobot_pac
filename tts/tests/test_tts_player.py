"""오디오 장치 없이 도는 TTSPlayer 테스트 (가짜 백엔드 사용).

실행: cd tts && python -m pytest -q tests
"""

import json
import threading
import time
import wave

import pytest

import tts_player
from phrases import CRITICAL_KEYS, PHRASES, load_phrases
from tts_player import DEFAULT_WAV_DIR, TTSPlayer


def write_wav(path, seconds=0.05, sr=16000, channels=1, width=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(b"\x00" * int(seconds * sr) * channels * width)


@pytest.fixture
def wav_dir(tmp_path):
    for k in PHRASES:
        write_wav(tmp_path / f"{k}.wav")
    return tmp_path


class FakeBackend:
    """호출 기록. gate 가 있으면 열릴 때까지 '재생' 중 상태로 막힌다."""

    def __init__(self, delay=0.0, fail=False, name="fake"):
        self.__name__ = name
        self.delay, self.fail = delay, fail
        self.calls = []
        self.gate = None
        self.started = threading.Event()

    def __call__(self, pcm, sr, device):
        self.calls.append((len(pcm), sr, device))
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("no audio device")


def make(wav_dir, *backends, **kw):
    return TTSPlayer(wav_dir=wav_dir, backends=backends, **kw)


# ---- 로딩 ----
def test_missing_wav_is_clear_error(wav_dir):
    (wav_dir / "done.wav").unlink()
    with pytest.raises(FileNotFoundError, match="done") as e:
        TTSPlayer(wav_dir=wav_dir, enabled=False)
    assert str(wav_dir) in str(e.value)


def test_stereo_wav_rejected(wav_dir):
    write_wav(wav_dir / "ready.wav", channels=2)
    with pytest.raises(ValueError, match="ready.wav"):
        TTSPlayer(wav_dir=wav_dir, enabled=False)


@pytest.mark.skipif(not all((DEFAULT_WAV_DIR / f"{k}.wav").is_file() for k in PHRASES),
                    reason="기본 wavs/ 가 없음")
def test_default_wavs_load():
    with TTSPlayer(enabled=False) as t:  # pour_rollout 과 같은 기본 사용법
        assert set(t.clips) == set(PHRASES)
        assert all(c.seconds > 0.3 for c in t.clips.values())
        assert t.critical_keys == CRITICAL_KEYS


# ---- say() ----
def test_say_returns_text_without_blocking(wav_dir):
    fake = FakeBackend(delay=0.5)
    with make(wav_dir, fake) as t:
        t0 = time.perf_counter()
        text = t.say("start")
        assert time.perf_counter() - t0 < 0.02
        assert text == PHRASES["start"]
        assert t.wait(timeout=3)
    assert t.played == [("start", "fake")]


def test_unknown_key_returns_empty(wav_dir, capsys):
    with make(wav_dir, FakeBackend()) as t:
        assert t.say("nope") == ""
    assert "nope" in capsys.readouterr().out


def test_disabled_never_plays(wav_dir):
    fake = FakeBackend()
    with make(wav_dir, fake, enabled=False) as t:
        assert t.say("done") == PHRASES["done"]
        assert t.wait(timeout=1)
    assert fake.calls == [] and t.played == []


def test_wait_timeout(wav_dir):
    fake = FakeBackend()
    fake.gate = threading.Event()
    t = make(wav_dir, fake)
    t.say("start")
    assert fake.started.wait(2)
    assert t.wait(timeout=0.05) is False
    fake.gate.set()
    assert t.wait(timeout=2) is True
    t.close()


def test_close_is_fast_while_playing(wav_dir):
    fake = FakeBackend()
    fake.gate = threading.Event()
    t = make(wav_dir, fake)
    t.say("start")
    t.say("ready")
    assert fake.started.wait(2)  # start 재생 중, ready 대기
    t0 = time.perf_counter()
    closer = threading.Thread(target=t.close)
    closer.start()
    while not t._stop:  # close() 가 대기열을 비운 뒤에
        time.sleep(0.001)
    fake.gate.set()  # 재생 끝
    closer.join(3)
    assert not closer.is_alive() and time.perf_counter() - t0 < 2.5
    assert not t._th.is_alive()
    assert [k for k, _ in t.played] == ["start"]  # 대기열은 버림


# ---- 큐 정책 ----
def test_plays_in_order(wav_dir):
    fake = FakeBackend()
    with make(wav_dir, fake, max_pending=5) as t:
        for k in ("ready", "start", "done"):
            t.say(k)
        assert t.wait(timeout=3)
    assert [k for k, _ in t.played] == ["ready", "start", "done"]


def test_duplicate_pending_key_is_ignored(wav_dir):
    fake = FakeBackend()
    fake.gate = threading.Event()
    with make(wav_dir, fake) as t:
        t.say("start")
        assert fake.started.wait(2)  # start 재생 중
        for _ in range(3):
            t.say("retry")
        fake.gate.set()
        assert t.wait(timeout=3)
    assert [k for k, _ in t.played] == ["start", "retry"]


def test_critical_never_dropped(wav_dir):
    fake = FakeBackend()
    fake.gate = threading.Event()
    with make(wav_dir, fake, max_pending=1) as t:
        t.say("start")
        assert fake.started.wait(2)
        for k in ("done", "ready", "retry", "stopped"):
            t.say(k)
        fake.gate.set()
        assert t.wait(timeout=3)
    played = [k for k, _ in t.played]
    assert played == ["start", "done", "stopped"]  # 일반 문구만 버려짐


def test_custom_critical_keys(wav_dir):
    fake = FakeBackend()
    fake.gate = threading.Event()
    with make(wav_dir, fake, max_pending=1, critical_keys={"ready"}) as t:
        t.say("start")
        assert fake.started.wait(2)
        t.say("ready")
        t.say("done")  # 이번엔 일반 문구
        fake.gate.set()
        assert t.wait(timeout=3)
    assert [k for k, _ in t.played] == ["start", "ready"]


# ---- 백엔드 실패 ----
def test_falls_back_to_next_backend(wav_dir):
    bad, good = FakeBackend(fail=True, name="bad"), FakeBackend(name="good")
    with make(wav_dir, bad, good) as t:
        t.say("start")
        t.say("done")
        assert t.wait(timeout=3)
        assert t.backend == "good"
    assert t.played == [("start", "good"), ("done", "good")]
    assert len(bad.calls) == 1  # 실패한 백엔드는 다시 시도하지 않음


def test_all_backends_fail_subtitle_only(wav_dir, capsys):
    bad = FakeBackend(fail=True)
    with make(wav_dir, bad) as t:
        assert t.say("start") == PHRASES["start"]
        assert t.wait(timeout=3)
        assert t.backend is None
        assert t.say("done") == PHRASES["done"]  # 이후에도 예외 없이 자막만
        assert t.wait(timeout=3)
    assert t.played == []
    out = capsys.readouterr().out
    assert out.count("자막만") == 1  # 경고는 한 번만


def test_missing_programs_subtitle_only(wav_dir):
    with make(wav_dir, "definitely-not-a-player-xyz") as t:
        assert t.backends == [] and t.backend is None
        assert t.say("ready") == PHRASES["ready"]
        assert t.wait(timeout=1)


def test_device_passed_to_backend(wav_dir):
    fake = FakeBackend()
    with make(wav_dir, fake, device="plughw:1,0") as t:
        t.say("start")
        assert t.wait(timeout=3)
    assert fake.calls[0][1:] == (16000, "plughw:1,0")


def test_command_line_includes_device():
    assert tts_player._cmd("aplay", 24000, "hw:1")[-3:] == ["-D", "hw:1", "-"]
    assert "--device=x" in tts_player._cmd("paplay", 24000, "x")


# ---- 사용자 문구 ----
def test_custom_phrases_mapping(tmp_path):
    write_wav(tmp_path / "hello.wav")
    with TTSPlayer(wav_dir=tmp_path, phrases={"hello": "안녕하세요"}, enabled=False) as t:
        assert t.say("hello") == "안녕하세요"
        assert t.say("done") == ""  # 기본 문구는 없음


def test_manifest_in_wav_dir_is_used(tmp_path):
    write_wav(tmp_path / "hi.wav")
    (tmp_path / "phrases.json").write_text(
        json.dumps({"phrases": {"hi": "Hi there"}, "critical": ["hi"]}), encoding="utf-8")
    with TTSPlayer(wav_dir=tmp_path, enabled=False) as t:
        assert t.say("hi") == "Hi there"
        assert t.critical_keys == {"hi"}


def test_load_phrases_py_and_flat_json(tmp_path):
    py = tmp_path / "my_phrases.py"
    py.write_text('PHRASES = {"a": "에이"}\nCRITICAL_KEYS = {"a"}\n', encoding="utf-8")
    assert load_phrases(py) == ({"a": "에이"}, frozenset({"a"}))
    js = tmp_path / "flat.json"
    js.write_text('{"b": "비"}', encoding="utf-8")
    assert load_phrases(js) == ({"b": "비"}, frozenset())
    with pytest.raises(ValueError):
        load_phrases(tmp_path / "x.txt")


def test_phrases_file_path_argument(tmp_path):
    js = tmp_path / "p.json"
    js.write_text('{"phrases": {"go": "출발"}}', encoding="utf-8")
    write_wav(tmp_path / "go.wav")
    with TTSPlayer(wav_dir=tmp_path, phrases=js, enabled=False) as t:
        assert t.say("go") == "출발"


# ---- 기타 ----
def test_output_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITA_LOCAL", str(tmp_path))
    assert tts_player.output_dir() == tmp_path / "outputs" / "tts"
    monkeypatch.delenv("UNITA_LOCAL")
    assert tts_player.output_dir().name in ("tts", "outputs")


def test_generate_wavs_postprocess_offline(tmp_path):
    """네트워크 없이 후처리·저장만 확인 (edge-tts 호출 안 함)."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    import generate_wavs as g

    sr = g.SR
    t = np.arange(sr) / sr
    x = np.concatenate([np.zeros(sr // 2), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(sr // 2)])
    y = g.postprocess(x.astype(np.float32))
    assert len(y) < len(x)  # 앞뒤 무음 제거
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    g.write_wav(tmp_path / "x.wav", y)
    with wave.open(str(tmp_path / "x.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, sr)


def test_generate_wavs_requires_out_with_phrases(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("av")
    import generate_wavs as g

    with pytest.raises(SystemExit):
        g.main(["--phrases", str(tmp_path / "p.json")])
