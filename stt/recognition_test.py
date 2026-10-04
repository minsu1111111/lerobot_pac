#!/usr/bin/env python3
"""인식률 테스트: wav 폴더 -> Whisper + 명령어 매칭 -> 정확도 / 혼동표 / 지연시간 / CSV.

파일 이름 규칙: <정답>__<아무거나>.wav,  정답 ∈ pour / stop / none
  예) pour__minsu_quiet_01.wav, stop__jiwon_noisy_03.wav, none__minsu_quiet_weather.wav
하위 폴더까지 찾는다. 결과 CSV: <출력 폴더>/recognition_<폴더명>_<시각>.csv
(출력 폴더는 voice_command.default_out_dir(): $UNITA_LOCAL/outputs/stt > ../../local/outputs/stt > stt/outputs)

    python stt/recognition_test.py CLIPS --device cpu
    python stt/recognition_test.py CLIPS --threads 4               # CPU 스레드 수 비교
    python stt/recognition_test.py CLIPS --initial-prompt ""     # 프롬프트 효과 비교
    python stt/recognition_test.py CLIPS --model base            # 모델 크기 비교

위험한 오류는 따로 센다: 정답 stop 을 놓침(stop 누락), 정답이 pour 가 아닌데 pour (오작동 붓기).
녹음은 stt/record_clips.py, 방법은 stt/README.md 참고.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_command as vcmd  # noqa: E402

LABELS = ("pour", "stop", "none")


def expected_label(path: Path) -> str | None:
    head = path.stem.split("__", 1)[0].lower()
    return head if head in LABELS and "__" in path.stem else None


def summarize(rows: list[dict]) -> dict:
    """rows(expected, got, stt_s) -> 정확도, 혼동표, 위험 오류, 지연시간."""
    conf = {e: {g: 0 for g in LABELS} for e in LABELS}
    for r in rows:
        conf[r["expected"]][r["got"]] += 1
    n = len(rows)
    correct = sum(conf[k][k] for k in LABELS)
    stt = np.array([r["stt_s"] for r in rows if r["stt_s"] > 0]) if rows else np.zeros(0)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "confusion": conf,
        "per_label_recall": {k: (conf[k][k] / sum(conf[k].values()) if sum(conf[k].values()) else None)
                             for k in LABELS},
        "missed_stop": conf["stop"]["pour"] + conf["stop"]["none"],
        "false_pour": conf["stop"]["pour"] + conf["none"]["pour"],
        "stt_mean_s": float(stt.mean()) if len(stt) else 0.0,
        "stt_median_s": float(np.median(stt)) if len(stt) else 0.0,
        "stt_p95_s": float(np.percentile(stt, 95)) if len(stt) else 0.0,
        "stt_max_s": float(stt.max()) if len(stt) else 0.0,
    }


def print_summary(s: dict) -> None:
    print(f"\n정확도: {s['accuracy'] * 100:.1f}%  ({round(s['accuracy'] * s['n'])}/{s['n']})")
    print("혼동표 (행 = 정답, 열 = 인식)")
    print("         " + "".join(f"{g:>7}" for g in LABELS))
    for e in LABELS:
        print(f"  {e:<7}" + "".join(f"{s['confusion'][e][g]:>7}" for g in LABELS))
    rec = ", ".join(f"{k} {v * 100:.0f}%" for k, v in s["per_label_recall"].items() if v is not None)
    print(f"라벨별 재현율: {rec}")
    print(f"위험 오류: stop 누락 {s['missed_stop']}건, 오작동 pour {s['false_pour']}건")
    print(f"STT 지연: 평균 {s['stt_mean_s']:.2f}s, 중앙값 {s['stt_median_s']:.2f}s, p95 {s['stt_p95_s']:.2f}s, 최대 {s['stt_max_s']:.2f}s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("clips", type=Path, help="wav 폴더")
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--initial-prompt", default=vcmd.INITIAL_PROMPT, help="'' = 사용 안 함")
    p.add_argument("--fallback", action="store_true", help="Whisper 기본 temperature 재시도 사용 (비교용)")
    p.add_argument("--sample-len", type=int, default=vcmd.SAMPLE_LEN, help="생성 토큰 상한 (0 = Whisper 기본 224)")
    p.add_argument("--threads", type=int, default=None, help="torch CPU 스레드 수 (기본: torch 기본)")
    p.add_argument("--out", type=Path, default=None, help="CSV 경로 (기본: <출력 폴더>/recognition_...csv)")
    args = p.parse_args(argv)
    vcmd.prefer_passive_omp()

    files = sorted(f for f in args.clips.rglob("*.wav") if expected_label(f))
    skipped = sorted(f.name for f in args.clips.rglob("*.wav") if not expected_label(f))
    if skipped:
        print(f"[건너뜀] 이름 규칙(<pour|stop|none>__*.wav)에 안 맞는 파일 {len(skipped)}개: {skipped[:5]}")
    if not files:
        print(f"[오류] {args.clips} 에 규칙에 맞는 wav가 없습니다.")
        return 1

    temp = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if args.fallback else 0.0
    vc = vcmd.VoiceCommander(args.model, args.device, initial_prompt=args.initial_prompt, temperature=temp,
                             sample_len=args.sample_len or None, num_threads=args.threads)  # 워밍업 포함

    rows = []
    print(f"\n{'결과':<4} {'정답':<5} {'인식':<5} {'STT(s)':>6}  파일 / 텍스트")
    for f in files:
        exp = expected_label(f)
        r = vc.transcribe_file(f)
        got = r.command or "none"
        rows.append({"file": str(f.relative_to(args.clips)), "expected": exp, "got": got,
                     "ok": int(exp == got), "text": r.text, "duration_s": round(r.duration_s, 2),
                     "stt_s": round(r.stt_s, 3)})
        print(f"{'O' if exp == got else 'X':<4} {exp:<5} {got:<5} {r.stt_s:>6.2f}  {f.name}  \"{r.text}\"",
              flush=True)

    s = summarize(rows)
    print_summary(s)

    out = args.out or vcmd.OUT_DIR / f"recognition_{args.clips.resolve().name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[저장] {out}  (model={args.model}, device={vc.device}, prompt={args.initial_prompt!r}, fallback={args.fallback}, "
          f"sample_len={args.sample_len}, threads={args.threads}, "
          f"로딩 {vc.load_s:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
