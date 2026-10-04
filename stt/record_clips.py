#!/usr/bin/env python3
"""실제 목소리 인식률 테스트 클립 녹음. 화면에 나오는 문장을 Enter 로 녹음 시작/종료.

시연과 같은 마이크, 같은 Enter 녹음 코드(voice_command.Microphone / record_until_enter)로 녹음해서
recognition_test.py 가 그대로 채점할 수 있는 이름으로 저장한다.

    파일 이름: <정답>__<화자>_<환경>_<NN>.wav   (정답 = pour / stop / none)
    예)       stop__minsu_noisy_03.wav

    python stt/record_clips.py --speaker minsu --condition quiet
    python stt/record_clips.py --speaker jiwon --condition noisy --repeat 2 --mic 3
    python stt/record_clips.py --list                       # 문장 목록만 보기
    python stt/recognition_test.py <출력 폴더>/real_clips --device cpu   # 채점

한 문장마다: Enter = 녹음 시작 -> 말하기 -> Enter = 녹음 끝 -> Enter = 저장 (r = 다시, s = 건너뜀, q = 종료).
같은 폴더에 이미 있는 번호는 건너뛰고 이어서 번호를 붙인다 (덮어쓰지 않음).
문장·화자·환경은 <출력 폴더>/manifest.csv 에도 한 줄씩 남는다.
녹음 프로토콜(화자 수, 거리, 소음, 합격 기준)은 stt/README.md 참고.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections.abc import Callable
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_command as vcmd  # noqa: E402

# 시연에서 실제로 나올 법한 말. pour/stop 은 팀 기본 명령("따라줘"/"정지")을 맨 앞에 둔다.
PHRASES: dict[str, list[str]] = {
    "pour": ["따라줘", "물 따라줘", "물 좀 따라 줄래?", "물 부어줘", "물 한 잔 줘", "물 좀 주세요"],
    "stop": ["정지", "멈춰", "그만", "스톱", "그만 따라", "잠깐 멈춰"],
    # 명령이 아닌 말 (오작동 확인용). "따라"/"물" 이 들어 있지만 명령이 아닌 문장도 넣는다.
    "none": ["안녕하세요", "오늘 날씨 좋네요", "나를 따라와", "물건 좀 집어줘", "이거 뭐예요?", "선물 좀 줘"],
}
CONDITIONS = ("quiet", "noisy")
CONDITION_HINT = {
    "quiet": "조용한 방. 마이크에서 약 1m, 시연 때와 같은 자세·목소리 크기로.",
    "noisy": "현장 소음 재생(부스 웅성거림·음악, 대략 60~70dB) 또는 옆에서 대화하는 상태. 마이크에서 약 1m.",
}
CLIP_RE = re.compile(r"^(?P<label>[a-z]+)__(?P<speaker>[^_]+)_(?P<cond>[^_]+)_(?P<idx>\d+)\.wav$")
MANIFEST = "manifest.csv"
MANIFEST_FIELDS = ["file", "expected", "phrase", "speaker", "condition", "duration_s", "peak", "timestamp"]
CLIP_PEAK = 32000  # 이 이상이면 클리핑 (입력 볼륨을 낮춰야 함)


def safe_name(value: str) -> str:
    """화자/환경 이름을 파일 이름에 넣을 수 있게. '_' 는 구분자라서 '-' 로 바꾼다."""
    name = re.sub(r"[^0-9A-Za-z가-힣-]+", "-", value.strip()).strip("-")
    if not name:
        raise ValueError(f"파일 이름으로 쓸 수 없는 이름: {value!r}")
    return name


def clip_name(label: str, speaker: str, condition: str, idx: int) -> str:
    return f"{label}__{speaker}_{condition}_{idx:02d}.wav"


def next_index(out: Path, label: str, speaker: str, condition: str) -> int:
    """out 에 이미 있는 같은 (정답, 화자, 환경) 클립 번호의 다음 번호."""
    used = [
        int(m["idx"])
        for f in out.glob(f"{label}__{speaker}_{condition}_*.wav")
        if (m := CLIP_RE.match(f.name))
    ]
    return max(used) + 1 if used else 0


def build_plan(
    phrases: dict[str, list[str]], labels: list[str], repeat: int, shuffle: bool, seed: int | None = None
) -> list[tuple[str, str]]:
    """[(정답, 문장), ...]. repeat 번 반복. shuffle 이면 라벨이 섞이도록 순서를 섞는다."""
    plan = [(label, text) for _ in range(repeat) for label in labels for text in phrases[label]]
    if shuffle:
        random.Random(seed).shuffle(plan)
    return plan


def pcm_stats(pcm: bytes) -> tuple[float, int]:
    x = np.frombuffer(pcm, dtype=np.int16)
    return len(x) / vcmd.SAMPLE_RATE, int(np.abs(x.astype(np.int32)).max()) if len(x) else 0


def record_once(mic, stdin, frame_ms: int, max_record_s: float) -> bytes:
    """Enter(또는 아무 줄)가 들어올 때까지 녹음. voice_command 의 Enter 모드와 같은 코드."""

    def stop_requested() -> bool:
        if vcmd.line_ready(stdin, 0.0):
            stdin.readline()
            return True
        return False

    with closing(mic.frames(frame_ms)) as frames:
        pcm, by_enter = vcmd.record_until_enter(frames, stop_requested, frame_ms, max_record_s)
    if not by_enter:
        vcmd.log(f"[녹음] 최대 길이 {max_record_s:g}초 도달 - 자동 종료")
    return pcm


def append_manifest(out: Path, row: dict) -> None:
    path = out / MANIFEST
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_session(
    plan: list[tuple[str, str]],
    mic,
    stdin,
    out: Path,
    speaker: str,
    condition: str,
    frame_ms: int = 30,
    max_record_s: float = 8.0,
    min_peak: int = vcmd.MIN_PEAK,
    ask: Callable[[str], None] = vcmd.log,
) -> list[dict]:
    """plan 을 차례로 녹음해 저장한다. 저장한 클립 정보 목록을 돌려준다. stdin EOF 또는 q 면 중단."""
    out.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for i, (label, text) in enumerate(plan, 1):
        while True:  # r(다시 녹음) 이면 같은 문장을 반복
            ask(f'\n[{i}/{len(plan)}] {label:<4}  "{text}"\n  Enter = 녹음 시작  |  s = 건너뜀  |  q = 종료')
            line = stdin.readline()
            cmd = line.strip().lower()
            if line == "" or cmd in vcmd.QUIT_KEYS:
                return saved
            if cmd == "s":
                break
            ask("[녹음] 말씀하세요... 끝나면 Enter")
            pcm = record_once(mic, stdin, frame_ms, max_record_s)
            duration, peak = pcm_stats(pcm)
            warn = ""
            if peak < min_peak:
                warn = "  <- 너무 작음 (마이크 연결/거리/음소거 확인)"
            elif peak >= CLIP_PEAK:
                warn = "  <- 클리핑 (입력 볼륨을 낮추세요)"
            ask(
                f"[녹음] {duration:.1f}초, 최대 진폭 {peak}{warn}\n"
                "  Enter = 저장  |  r = 다시 녹음  |  s = 건너뜀  |  q = 종료"
            )
            line = stdin.readline()
            cmd = line.strip().lower()
            if cmd == "r":
                continue
            if line == "" or cmd in vcmd.QUIT_KEYS:
                return saved
            if cmd == "s":
                break
            name = clip_name(label, speaker, condition, next_index(out, label, speaker, condition))
            vcmd.save_wav(out / name, pcm)
            row = {
                "file": name,
                "expected": label,
                "phrase": text,
                "speaker": speaker,
                "condition": condition,
                "duration_s": round(duration, 2),
                "peak": peak,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            append_manifest(out, row)
            saved.append(row)
            ask(f"[저장] {name}")
            break
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--speaker", help="화자 이름 (영문/한글, 예: minsu)")
    p.add_argument("--condition", choices=CONDITIONS, help="quiet = 조용함, noisy = 현장 소음")
    p.add_argument("--repeat", type=int, default=1, help="문장 목록을 몇 번 반복할지")
    p.add_argument("--labels", default="pour,stop,none", help="녹음할 정답 종류 (쉼표로)")
    p.add_argument("--out", type=Path, default=vcmd.OUT_DIR / "real_clips", help="저장 폴더")
    p.add_argument("--mic", type=vcmd.parse_device, default=None, help="입력 장치 번호 또는 이름 일부")
    p.add_argument("--max-record-s", type=float, default=8.0, help="한 문장 최대 녹음 길이(초)")
    p.add_argument("--frame-ms", type=int, default=30, choices=(10, 20, 30))
    p.add_argument("--no-shuffle", action="store_true", help="문장을 목록 순서대로 (기본: 섞음)")
    p.add_argument("--seed", type=int, default=None, help="섞는 순서 고정용")
    p.add_argument("--list", action="store_true", help="문장 목록만 출력하고 종료")
    p.add_argument("--list-devices", action="store_true", help="오디오 장치 목록 출력 후 종료")
    args = p.parse_args(argv)
    args.labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    bad = [x for x in args.labels if x not in PHRASES]
    if bad:
        p.error(f"--labels 는 {list(PHRASES)} 중에서: {bad}")
    if not (args.list or args.list_devices):
        if not args.speaker or not args.condition:
            p.error("--speaker 와 --condition 이 필요합니다")
        try:
            args.speaker = safe_name(args.speaker)
        except ValueError as e:
            p.error(str(e))
    if args.repeat < 1 or args.max_record_s <= 0:
        p.error("--repeat 는 1 이상, --max-record-s 는 0보다 커야 합니다")
    return args


def main(argv: list[str] | None = None) -> int:
    vcmd.configure_console()
    args = parse_args(argv)
    if args.list:
        for label in args.labels:
            print(f"{label}: " + " / ".join(PHRASES[label]))
        return vcmd.EXIT_OK
    try:
        if args.list_devices:
            print(vcmd.import_sounddevice().query_devices())
            return vcmd.EXIT_OK
        mic = vcmd.open_microphone(args.mic)
    except vcmd.SttError as e:
        vcmd.log(str(e))
        return vcmd.EXIT_ERROR

    plan = build_plan(PHRASES, args.labels, args.repeat, not args.no_shuffle, args.seed)
    vcmd.log(
        f"[녹음 준비] 화자 {args.speaker}, 환경 {args.condition}, {len(plan)}문장 -> {args.out}\n"
        f"  {CONDITION_HINT[args.condition]}"
    )
    try:
        saved = run_session(
            plan, mic, sys.stdin, args.out, args.speaker, args.condition, args.frame_ms, args.max_record_s
        )
    except KeyboardInterrupt:
        vcmd.log("\n[종료] Ctrl+C (그때까지 저장한 클립은 남아 있음)")
        return vcmd.EXIT_OK
    vcmd.log(
        f"\n[완료] {len(saved)}개 저장 -> {args.out}\n"
        f"  채점: python {Path(__file__).with_name('recognition_test.py')} {args.out} --device cpu"
    )
    return vcmd.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
