"""record_clips.py 로직 테스트 (가짜 마이크 + 가짜 키 입력, 마이크·PortAudio 없이 돈다)."""

import csv
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recognition_test as rt  # noqa: E402

import record_clips as rc  # noqa: E402
import voice_command as v  # noqa: E402


class FakeMic:
    peak = 10000

    def __init__(self, amp=8000):
        self.amp, self.opened = amp, 0

    def frames(self, frame_ms):
        self.opened += 1
        n = v.SAMPLE_RATE * frame_ms // 1000
        while True:
            yield np.full(n, self.amp, np.int16).tobytes()


def session(keys, plan, out, amp=8000, **kw):
    msgs = []
    mic = FakeMic(amp)
    saved = rc.run_session(plan, mic, io.StringIO(keys), out, "minsu", "quiet", ask=msgs.append, **kw)
    return saved, msgs, mic


def test_names_and_index():
    assert rc.clip_name("stop", "minsu", "noisy", 3) == "stop__minsu_noisy_03.wav"
    assert rc.safe_name(" Kim_Min su ") == "Kim-Min-su"
    assert rc.safe_name("지원") == "지원"
    try:
        rc.safe_name("__")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        assert rc.next_index(out, "pour", "minsu", "quiet") == 0
        for name in (
            "pour__minsu_quiet_00.wav",
            "pour__minsu_quiet_04.wav",
            "pour__minsu_noisy_09.wav",
            "stop__minsu_quiet_07.wav",
            "pour__jiwon_quiet_08.wav",
        ):
            (out / name).touch()
        assert rc.next_index(out, "pour", "minsu", "quiet") == 5


def test_build_plan():
    plan = rc.build_plan(rc.PHRASES, ["pour", "stop"], 2, shuffle=False)
    assert len(plan) == 2 * (len(rc.PHRASES["pour"]) + len(rc.PHRASES["stop"]))
    assert plan[0] == ("pour", "따라줘") and ("stop", "정지") in plan
    shuffled = rc.build_plan(rc.PHRASES, ["pour", "stop", "none"], 1, shuffle=True, seed=1)
    assert sorted(shuffled) == sorted(rc.build_plan(rc.PHRASES, ["pour", "stop", "none"], 1, shuffle=False))
    assert shuffled == rc.build_plan(rc.PHRASES, ["pour", "stop", "none"], 1, shuffle=True, seed=1)


def test_phrases_match_their_labels():
    # 문장 목록 자체가 명령어 사전과 맞아야 채점이 의미가 있다.
    for label, phrases in rc.PHRASES.items():
        for text in phrases:
            assert (v.match_command(text) or "none") == label, (label, text)


def test_session_saves_clips_scoreable_by_recognition_test():
    plan = [("pour", "물 따라줘"), ("stop", "정지"), ("none", "안녕하세요")]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "clips"
        # 한 문장마다: Enter(시작) Enter(끝) Enter(저장)
        saved, _, mic = session("\n\n\n" * 3, plan, out)
        assert [r["file"] for r in saved] == [
            "pour__minsu_quiet_00.wav",
            "stop__minsu_quiet_00.wav",
            "none__minsu_quiet_00.wav",
        ]
        assert mic.opened == 3  # 녹음마다 스트림을 새로 연다
        for r in saved:
            f = out / r["file"]
            assert rt.expected_label(f) == r["expected"]
            x = v.load_wav(f)
            assert len(x) > 0 and abs(x.max() - 8000 / 32768) < 1e-3
            assert r["peak"] == 8000 and r["duration_s"] > 0
        with open(out / rc.MANIFEST, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [(r["file"], r["phrase"]) for r in rows] == [(s["file"], s["phrase"]) for s in saved]
        # 두 번째 세션은 번호를 이어 붙이고 manifest 에 덧붙인다
        saved2, _, _ = session("\n\n\n", plan[:1], out)
        assert saved2[0]["file"] == "pour__minsu_quiet_01.wav"
        with open(out / rc.MANIFEST, encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == 4


def test_session_redo_skip_quit():
    plan = [("pour", "물 따라줘"), ("stop", "정지"), ("none", "안녕하세요"), ("stop", "멈춰")]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        keys = (
            "\n\nr\n"
            "\n\n\n"  # 1: 녹음 -> r(다시) -> 녹음 -> 저장
            "s\n"  # 2: 녹음 전에 건너뜀
            "\n\ns\n"  # 3: 녹음 후 건너뜀
            "q\n"
        )  # 4: 종료
        saved, _, mic = session(keys, plan, out)
        assert [r["file"] for r in saved] == ["pour__minsu_quiet_00.wav"]
        assert mic.opened == 3
        assert sorted(p.name for p in out.glob("*.wav")) == ["pour__minsu_quiet_00.wav"]


def test_session_eof_and_quiet_warning():
    with tempfile.TemporaryDirectory() as d:
        saved, msgs, _ = session("\n\n", [("stop", "정지")], Path(d), amp=10)  # 저장 확인 전에 EOF
        assert saved == []
        assert any("너무 작음" in m for m in msgs)
        saved, msgs, _ = session("\n\n\n", [("stop", "정지")], Path(d), amp=32767)
        assert len(saved) == 1 and any("클리핑" in m for m in msgs)


def test_parse_args():
    a = rc.parse_args(["--speaker", "min_su", "--condition", "noisy", "--repeat", "2", "--mic", "3"])
    assert (a.speaker, a.condition, a.repeat, a.mic) == ("min-su", "noisy", 2, 3)
    assert a.out == v.OUT_DIR / "real_clips" and a.labels == ["pour", "stop", "none"]
    assert rc.parse_args(["--list"]).list
    for bad in (
        [],
        ["--speaker", "a"],
        ["--speaker", "a", "--condition", "loud"],
        ["--speaker", "a", "--condition", "quiet", "--labels", "pour,drink"],
        ["--speaker", "a", "--condition", "quiet", "--repeat", "0"],
    ):
        try:
            rc.parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_list_runs_without_microphone():
    assert rc.main(["--list"]) == 0


if __name__ == "__main__":
    import contextlib

    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} passed")
