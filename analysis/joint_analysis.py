"""할 일 1: 오른팔 관절 분석 + 완료 감지기 오프라인 검증.

데이터셋 parquet만 읽는다(영상 디코딩 없음, 원본 수정 없음).
  1) 오른팔 6관절을 100 에피소드 겹쳐 그려 붓기 관절 선택 근거를 만든다.
  2) 선택 관절의 기준선(처음 base_sec 중앙값)과 최대 기울기 진폭으로 tilt/return 임계값 계산.
  3) 100 에피소드에 감지기를 돌려 감지율·감지 시점을 낸다 (state, action 둘 다).
  4) 임계값/hold 민감도 표.

실행:
  python joint_analysis.py                 # 기본값
  python joint_analysis.py --tilt-frac 0.6 --return-frac 0.3 --hold 5
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASET, OUTPUTS, THRESHOLDS  # noqa: E402
from pour_detector import DONE, PourDetector, PourDetectorConfig  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_STATE, C_ACTION, C_TILT, C_RET = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"


def load(ds: Path):
    info = json.loads((ds / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    files = sorted(glob.glob(str(ds / "data/*/*.parquet")))
    df = pd.concat(
        [pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "observation.state", "action"]) for f in files]
    ).sort_values("index")
    eps = {}
    for e, g in df.groupby("episode_index", sort=True):
        assert (np.diff(g["frame_index"].values) == 1).all(), f"ep{e}: frame_index 불연속"
        eps[int(e)] = {"state": np.stack(g["observation.state"].values), "action": np.stack(g["action"].values)}
    return info, names, eps


def excursions(x, base, thr):
    """기준선 대비 |편차| > thr 구간 개수 (붓기 관절이면 1이어야 함)."""
    above = np.abs(x - base) > thr
    return int(np.sum(np.diff(above.astype(int)) == 1) + above[0])


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--joint", default="right_wrist_roll.pos")
    ap.add_argument("--tilt-frac", type=float, default=0.6)
    ap.add_argument("--return-frac", type=float, default=0.3)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--base-sec", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "joint_analysis")
    ap.add_argument("--install", action="store_true", help=f"결과 임계값을 롤아웃 기본값({THRESHOLDS.name})으로 저장")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base_n = int(args.base_sec * FPS)

    info, names, eps = load(args.dataset)
    assert info["fps"] == FPS
    ji = names.index(args.joint)
    print(f"dataset: {len(eps)} eps, {sum(len(v['state']) for v in eps.values())} frames, joint={args.joint} (idx {ji})")

    # ---- 1) 관절 선택 근거: 오른팔 6관절 진폭 / 큰 움직임 횟수 ----
    print("\n[관절 비교] 기준선=처음 %.1fs 중앙값, 진폭=|편차| 최대, 횟수=|편차|>진폭50%% 구간 수" % args.base_sec)
    print(f"{'joint':16s} {'진폭 중앙값':>10s} {'횟수=1 비율':>11s} {'횟수 중앙값':>10s}")
    joint_rows = []
    for j, n in enumerate(JOINTS):
        col = 6 + j
        amps, exc = [], []
        for v in eps.values():
            x = v["state"][:, col]
            b = np.median(x[:base_n])
            a = np.abs(x - b).max()
            amps.append(a)
            exc.append(excursions(x, b, 0.5 * a))
        exc = np.array(exc)
        joint_rows.append((n, np.median(amps), (exc == 1).mean(), np.median(exc)))
        print(f"right_{n:10s} {np.median(amps):10.1f} {(exc == 1).mean():11.0%} {np.median(exc):10.0f}")

    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, (ax, n) in enumerate(zip(axs.flat, JOINTS)):
        for v in eps.values():
            x = v["state"][:, 6 + j]
            ax.plot(np.arange(len(x)) / FPS, x, lw=0.6, color=C_STATE, alpha=0.12)
        r = joint_rows[j]
        ax.set_title(f"right_{n}   진폭 {r[1]:.0f} · 큰 움직임 1회 {r[2]:.0%}", fontsize=10, color=INK, loc="left")
        style(ax)
    for ax in axs[-1]:
        ax.set_xlabel("시간 (s)", color=INK2)
    fig.suptitle("오른팔 6관절 · observation.state · 100 에피소드 겹침", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(args.out / "joint_overview.png", dpi=110)
    plt.close(fig)

    # ---- 2) 기준선·진폭·임계값 ----
    per = {}
    for e, v in eps.items():
        x = v["state"][:, ji]
        b = float(np.median(x[:base_n]))
        d = x - b
        pk = int(np.argmax(np.abs(d)))
        per[e] = dict(base=b, peak_dev=float(d[pk]), peak_t=pk / FPS, len_s=len(x) / FPS)
    peak_devs = np.array([p["peak_dev"] for p in per.values()])
    sign = float(np.sign(np.median(peak_devs)))
    amp = float(np.median(sign * peak_devs))
    tilt_off, return_off = args.tilt_frac * amp, args.return_frac * amp
    bases = np.array([p["base"] for p in per.values()])
    print(f"\n[임계값] sign={sign:+.0f}  진폭(에피소드별 최대편차의 중앙값)={amp:.1f}")
    print(f"  기준선: 중앙값 {np.median(bases):.1f}, 범위 {bases.min():.1f} ~ {bases.max():.1f}")
    print(f"  에피소드별 최대편차: 최소 {(sign * peak_devs).min():.1f}, p5 {np.percentile(sign * peak_devs, 5):.1f}, 최대 {(sign * peak_devs).max():.1f}")
    print(f"  tilt_off   = {args.tilt_frac:.2f} x {amp:.1f} = {tilt_off:.1f}  (절대값 환산 ≈ {np.median(bases) + sign * tilt_off:.1f})")
    print(f"  return_off = {args.return_frac:.2f} x {amp:.1f} = {return_off:.1f}  (절대값 환산 ≈ {np.median(bases) + sign * return_off:.1f})")

    # ---- 3) 감지기 실행 (state / action) ----
    cfg = PourDetectorConfig(tilt_off=tilt_off, return_off=return_off, hold=args.hold, base_frames=base_n, sign=sign)

    def run_all(src, c):
        rows = []
        for e, v in eps.items():
            x = v[src][:, ji]
            det = PourDetector(c).run(x)
            n_tilt = excursions(x, det.baseline, tilt_off) if det.baseline is not None else 0
            rows.append(
                dict(
                    episode=e,
                    detected=det.state == DONE,
                    final_state=det.state,
                    tilt_t=None if det.tilt_step is None else det.tilt_step / FPS,
                    done_t=None if det.done_step is None else det.done_step / FPS,
                    peak_t=per[e]["peak_t"],
                    len_s=per[e]["len_s"],
                    n_tilt_crossings=n_tilt,
                )
            )
        r = pd.DataFrame(rows)
        r["done_after_peak_s"] = r["done_t"] - r["peak_t"]
        r["end_after_done_s"] = r["len_s"] - r["done_t"]
        r["pour_s"] = r["done_t"] - r["tilt_t"]
        return r

    res = {src: run_all(src, cfg) for src in ("state", "action")}
    summary = dict(
        joint=args.joint, sign=sign, amp=amp, tilt_frac=args.tilt_frac, return_frac=args.return_frac,
        tilt_off=tilt_off, return_off=return_off, hold=args.hold, base_frames=base_n,
        baseline_median=float(np.median(bases)),
    )
    for src, r in res.items():
        d = r[r.detected]
        print(f"\n[감지 결과 · {src}]  감지율 {r.detected.sum()}/{len(r)}"
              f"   tilt 경계 다중 통과 에피소드 {int((r.n_tilt_crossings > 1).sum())}")
        if len(r) - len(d):
            print("  미감지:", r[~r.detected][["episode", "final_state"]].to_dict("records"))
        for col, lab in [("tilt_t", "POURING 진입 (에피소드 시작 기준 s)"), ("done_t", "DONE (s)"),
                         ("pour_s", "POURING 유지 시간 (s)"), ("done_after_peak_s", "DONE - 최대 기울기 시점 (s)"),
                         ("end_after_done_s", "DONE 후 에피소드 종료까지 (s)")]:
            q = d[col].quantile([0, 0.5, 1]).values
            print(f"  {lab:34s} 최소 {q[0]:6.1f}  중앙 {q[1]:6.1f}  최대 {q[2]:6.1f}")
        summary[src] = dict(
            detected=int(r.detected.sum()), n=len(r), multi_crossing=int((r.n_tilt_crossings > 1).sum()),
            done_t_max=float(d.done_t.max()), done_t_median=float(d.done_t.median()),
            end_after_done_median=float(d.end_after_done_s.median()),
        )
        r.to_csv(args.out / f"detections_{src}.csv", index=False)

    to_max = summary["state"]["done_t_max"]
    summary["timeout_suggest_s"] = float(np.ceil(to_max * 1.3))
    print(f"\n[타임아웃 백업 제안] 데이터 최대 DONE {to_max:.1f}s × 1.3 ≈ {summary['timeout_suggest_s']:.0f}s "
          f"(= {int(summary['timeout_suggest_s'] * FPS)} step @30fps)")

    # ---- 4) 민감도 ----
    print("\n[민감도 · state]  감지율 / tilt 다중통과 에피소드 수 / DONE-최대기울기 중앙값(s)")
    sweep = []
    for tf in (0.4, 0.5, 0.6, 0.7, 0.8):
        for rf in (0.1, 0.2, 0.3, 0.4, 0.5):
            if rf >= tf:
                continue
            for h in (1, 5, 10, 15):
                c = PourDetectorConfig(tilt_off=tf * amp, return_off=rf * amp, hold=h, base_frames=base_n, sign=sign)
                ok, lat = 0, []
                for e, v in eps.items():
                    det = PourDetector(c).run(v["state"][:, ji])
                    if det.state == DONE:
                        ok += 1
                        lat.append(det.done_step / FPS - per[e]["peak_t"])
                multi = sum(excursions(v["state"][:, ji], np.median(v["state"][:base_n, ji]), tf * amp) > 1 for v in eps.values())
                sweep.append(dict(tilt_frac=tf, return_frac=rf, hold=h, detected=ok, multi_crossing=multi,
                                  done_after_peak_med=float(np.median(lat)) if lat else None))
    sw = pd.DataFrame(sweep)
    sw.to_csv(args.out / "sweep_state.csv", index=False)
    piv = sw.pivot_table(index=["tilt_frac", "return_frac"], columns="hold", values="detected")
    print(piv.to_string())

    # ---- 그림: 에피소드별 small multiples ----
    order = sorted(eps)
    nc = 10
    nr = int(np.ceil(len(order) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.5 * nr), sharey=True)
    rs, ra = res["state"].set_index("episode"), res["action"].set_index("episode")
    for ax, e in zip(axs.flat, order):
        b = per[e]["base"]
        t = np.arange(len(eps[e]["state"])) / FPS
        ax.plot(t, eps[e]["action"][:, ji], lw=0.8, color=C_ACTION)
        ax.plot(t, eps[e]["state"][:, ji], lw=1.0, color=C_STATE)
        ax.axhline(b + sign * tilt_off, color=C_TILT, lw=0.6, ls="--")
        ax.axhline(b + sign * return_off, color=C_RET, lw=0.6, ls="--")
        row = rs.loc[e]
        if row.detected:
            ax.axvline(row.done_t, color=INK, lw=0.8)
        ax.set_title(f"ep{e}" + ("" if row.detected else " 미감지"), fontsize=7, color=INK if row.detected else C_TILT, pad=2)
        ax.tick_params(labelsize=6, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axs.flat)[len(order):]:
        ax.axis("off")
    fig.suptitle(f"{args.joint} · 파랑 state / 주황 action · 빨강 점선 tilt, 초록 점선 return · 검정 세로선 DONE(state)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out / "wrist_roll_all_episodes.png", dpi=90)
    plt.close(fig)

    # ---- 그림: DONE 시점 정렬 ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for e in order:
        row = rs.loc[e]
        if not row.detected:
            continue
        x = sign * (eps[e]["state"][:, ji] - per[e]["base"])
        t = np.arange(len(x)) / FPS - row.done_t
        ax.plot(t, x, lw=0.6, color=C_STATE, alpha=0.25)
    ax.axhline(tilt_off, color=C_TILT, lw=1, ls="--")
    ax.axhline(return_off, color=C_RET, lw=1, ls="--")
    ax.text(ax.get_xlim()[0], tilt_off, f" tilt_off {tilt_off:.0f}", color=INK2, va="bottom", fontsize=8)
    ax.text(ax.get_xlim()[0], return_off, f" return_off {return_off:.0f}", color=INK2, va="bottom", fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("DONE 기준 시간 (s)", color=INK2)
    ax.set_ylabel("기준선 대비 편차", color=INK2)
    ax.set_title(f"{args.joint} · state · DONE 시점 정렬 ({int(rs.detected.sum())} 에피소드)", loc="left", color=INK, fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(args.out / "wrist_roll_aligned_done.png", dpi=110)
    plt.close(fig)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.install:
        THRESHOLDS.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"롤아웃 임계값 갱신: {THRESHOLDS}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
