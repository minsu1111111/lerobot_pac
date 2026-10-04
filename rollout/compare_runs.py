"""pour_rollout.py 결과(summary.json) 여러 개를 표로 비교한다 (청크 점프 완화책 비교용).

실행:
  python compare_runs.py                     # local/outputs/rollout/* 전부
  python compare_runs.py --glob "*_fast_*"   # 일부
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OUTPUTS  # noqa: E402


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k)
    return d


def f(x, n=1):
    return "-" if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=OUTPUTS / "rollout")
    ap.add_argument("--glob", default="*")
    a = ap.parse_args()
    hdr = (
        "세션",
        "ep",
        "n_act",
        "TE",
        "clamp",
        "정지",
        "DONE",
        "시연DONE",
        "cmdDONE",
        "경계max",
        "경계p95",
        "내부max",
        "팔p99",
        "roll max",
        "grip max",
        "지연",
        "청크추론p50ms",
        "주기p95ms",
    )
    rows = []
    for p in sorted(a.root.glob(a.glob)):
        sj = p / "summary.json"
        if not sj.is_file():
            continue
        s = json.loads(sj.read_text())
        for r in s.get("runs", []):
            rows.append(
                (
                    p.name[16:] or p.name,
                    str(r.get("episode", "-")),
                    str(s.get("n_action_steps")),
                    str(s.get("temporal_ensemble") or "-"),
                    str(g(s, "args", "max_step_deg") or "-"),
                    r["stop_reason"],
                    f(r["done_t"]),
                    f(r.get("demo_done_t")),
                    f(g(r, "cmd_detector", "done_t")),
                    f(g(r, "jump_arm_boundary", "max")),
                    f(g(r, "jump_arm_boundary", "p95")),
                    f(g(r, "jump_arm_within", "max")),
                    f(g(r, "jump_arm_all", "p99")),
                    f(g(r, "jump_roll_all", "max")),
                    f(g(r, "jump_grip_all", "max")),
                    str(r["overruns"]),
                    f(g(r, "infer_ms_chunk", "p50"), 0),
                    f(g(r, "period_ms", "p95"), 1),
                )
            )
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)] if rows else [len(h) for h in hdr]
    for r in [hdr, *rows]:
        print("  ".join(str(x).ljust(n) for x, n in zip(r, w)))


if __name__ == "__main__":
    main()
