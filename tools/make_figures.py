"""README 용 그림·영상 (docs/media/) 다시 만들기.

깃에 안 올리는 결과물(local/outputs)과 데이터셋에서 읽는다. 원본은 수정하지 않는다.
  detector_aligned.png  오른손목 roll 을 DONE 시점에 정렬 (100 에피소드) + tilt/return 임계값
  joint_selection.png   오른팔 6관절 겹침 → wrist_roll 을 고른 근거 (큰 움직임 1회 비율)
  model_vs_demo.png     대표 에피소드 1개: ACT open-loop 출력 vs 시연 action
  sim_stills.png        시뮬 롤아웃 영상 4장면 (잡기 / 붓기 최대 / 완료 감지+안내 / 내려놓기)
  demo_sim.gif          시뮬 롤아웃 요약 루프 (느린 구간 빨리 감기, 완료 안내 구간은 실제 속도)
  demo_sim.mp4          시뮬 롤아웃 전체 + TTS 소리 (저용량 재인코딩)

필요한 입력:
  validate/offline_model.py  → outputs/offline_model/{actions,offline}_stride100.*
  rollout/pour_rollout.py --backend replay+mujoco --episode 25 --sim-video ...  → outputs/sim/rollout_ep25_sim_audio.mp4

실행:
  python tools/make_figures.py                   # 전부 → docs/media
  python tools/make_figures.py --skip-video      # PNG 만
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.joint_analysis import FPS, JOINTS, excursions, load  # noqa: E402
from paths import DATASET, GITHUB, OUTPUTS, THRESHOLDS  # noqa: E402
from pour_detector import DONE, PourDetector, PourDetectorConfig  # noqa: E402

plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK KR", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    }
)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8b85", "#ebeae6"
C_DEMO, C_MODEL, C_TILT, C_RET = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"
MAX_PNG = 600_000
SIM_VIDEO = OUTPUTS / "sim" / "rollout_ep25_sim_audio.mp4"
SIM_RUN = OUTPUTS / "rollout" / "20261004_114144_ep25_sim_video_full"


def style(ax, size=9):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=size, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)


def save(fig, path: Path, dpi=100):
    """PNG 저장. MAX_PNG 를 넘으면 256색으로 줄인다."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    if buf.tell() > MAX_PNG:
        im = (
            Image.open(buf)
            .convert("RGB")
            .quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        )
        buf = io.BytesIO()
        im.save(buf, format="png", optimize=True)
    path.write_bytes(buf.getvalue())
    print(f"  {path.name:22s} {path.stat().st_size / 1e3:6.0f} KB")


def detector_cfg(th: dict) -> PourDetectorConfig:
    return PourDetectorConfig(
        tilt_off=th["tilt_off"],
        return_off=th["return_off"],
        hold=th["hold"],
        base_frames=th["base_frames"],
        sign=th["sign"],
    )


# ---------------------------------------------------------------- 데이터셋 그림
def fig_detector_aligned(eps, ji, th, out: Path):
    cfg = detector_cfg(th)
    fig, ax = plt.subplots(figsize=(16, 7))
    grid = np.arange(-40 * FPS, 15 * FPS) / FPS
    stack, n_det = [], 0
    for v in eps.values():
        x = v["state"][:, ji]
        det = PourDetector(cfg).run(x)
        if det.state != DONE:
            continue
        n_det += 1
        d = cfg.sign * (x - det.baseline)
        t = np.arange(len(x)) / FPS - det.done_step / FPS
        ax.plot(t, d, lw=0.7, color=C_DEMO, alpha=0.16)
        stack.append(np.interp(grid, t, d, left=np.nan, right=np.nan))
    stack = np.array(stack)
    ok = np.sum(~np.isnan(stack), 0) >= len(stack) // 2
    med = np.where(ok, np.nanmedian(stack, 0), np.nan)
    ax.plot(grid, med, lw=2.0, color=C_DEMO, label="중앙값 (median)")

    x0, x1 = -32, 12
    ax.axhline(cfg.tilt_off, color=C_TILT, lw=1.3, ls=(0, (5, 3)))
    ax.axhline(cfg.return_off, color=C_RET, lw=1.3, ls=(0, (5, 3)))
    ax.axvline(0, color=INK, lw=1.0)
    kw = dict(color=INK, fontsize=11, va="bottom", ha="right")
    ax.text(
        x1 - 0.3,
        cfg.tilt_off + 2,
        f"tilt +{cfg.tilt_off:.1f}° · {cfg.hold}프레임 넘으면 → 붓는 중 (POURING)",
        **kw,
    )
    ax.text(
        x1 - 0.3,
        cfg.return_off + 2,
        f"return +{cfg.return_off:.1f}° · {cfg.hold}프레임 들어오면 → 완료 (DONE)",
        **kw,
    )
    ax.text(0.3, 162, 'DONE → "다 따랐습니다" (TTS)', color=INK, fontsize=11, va="top")
    ax.set_xlim(x0, x1)
    ax.set_ylim(-15, 165)
    ax.set_xlabel("DONE 기준 시간 (s)", color=INK2, fontsize=11)
    ax.set_ylabel("기준선 대비 편차 (°)", color=INK2, fontsize=11)
    style(ax, 10)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK2)
    fig.suptitle(
        f"완료 감지: 오른손목 roll 히스테리시스 ({n_det}/{len(eps)} 감지)",
        x=0.012,
        y=0.975,
        ha="left",
        color=INK,
        fontsize=17,
    )
    fig.text(
        0.012,
        0.915,
        f"right_wrist_roll · observation.state · {len(eps)} 에피소드를 DONE 시점에 정렬 · "
        f"기준선 = 처음 {cfg.base_frames / FPS:.0f} s 중앙값 · 임계값 = 진폭 중앙값 {th['amp']:.0f}° × "
        f"{th['tilt_frac']:.1f} / {th['return_frac']:.1f}",
        color=INK2,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.865, bottom=0.085)
    save(fig, out / "detector_aligned.png")


def fig_joint_selection(eps, base_n, out: Path):
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, (ax, n) in enumerate(zip(axs.flat, JOINTS)):
        col = 6 + j
        amps, exc = [], []
        pick = n == "wrist_roll"
        for v in eps.values():
            x = v["state"][:, col]
            b = np.median(x[:base_n])
            a = np.abs(x - b).max()
            amps.append(a)
            exc.append(excursions(x, b, 0.5 * a))
            ax.plot(
                np.arange(len(x)) / FPS,
                x,
                lw=0.6,
                color=C_DEMO if pick else MUTED,
                alpha=0.14 if pick else 0.12,
            )
        once = np.mean(np.array(exc) == 1)
        ax.set_title(
            f"right_{n}", loc="left", fontsize=13, color=INK, fontweight="bold" if pick else "normal", pad=22
        )
        ax.text(
            0,
            1.03,
            f"큰 움직임 1회 {once:.0%} · 진폭 중앙값 {np.median(amps):.0f}°" + ("  ← 선택" if pick else ""),
            transform=ax.transAxes,
            fontsize=11,
            color=INK if pick else INK2,
        )
        style(ax, 9)
    for ax in axs[-1]:
        ax.set_xlabel("시간 (s)", color=INK2, fontsize=10)
    for ax in axs[:, 0]:
        ax.set_ylabel("관절 값 (°, gripper 0–100)", color=INK2, fontsize=10)
    fig.suptitle(
        "왜 wrist_roll 인가: 오른팔(물통) 6관절 × 100 에피소드",
        x=0.012,
        y=0.98,
        ha="left",
        color=INK,
        fontsize=17,
    )
    fig.text(
        0.012,
        0.925,
        "큰 움직임 = 기준선 대비 |편차| 가 그 에피소드 최대 진폭의 50% 를 넘는 구간. "
        "1회 비율이 높을수록 '붓기 1번 = 한 번 크게 기울였다 돌아옴' 과 1:1 로 맞는다.",
        color=INK2,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.07, hspace=0.42, wspace=0.12)
    save(fig, out / "joint_selection.png")


def fig_model_vs_demo(th, out: Path, episode=None):
    off = OUTPUTS / "offline_model"
    r = pd.read_csv(off / "offline_stride100.csv")
    if episode is None:  # MAE 중앙값에 가장 가까운 에피소드
        episode = int(r.iloc[(r.mae_all - r.mae_all.median()).abs().argsort().iloc[0]]["episode"])
    z = np.load(off / "actions_stride100.npz", allow_pickle=True)
    names = [str(n).removesuffix(".pos") for n in z["names"]]
    demo, pred = z[f"ep{episode}_demo"], z[f"ep{episode}_model"]
    cfg = detector_cfg(th)
    ji = names.index(th["joint"].removesuffix(".pos"))
    dd, dm = PourDetector(cfg).run(demo[:, ji]), PourDetector(cfg).run(pred[:, ji])
    t = np.arange(len(demo)) / FPS
    row = r.set_index("episode").loc[episode]

    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.5, 1), hspace=0.32, wspace=0.15)
    panels = [(fig.add_subplot(gs[0, :]), ji)] + [
        (fig.add_subplot(gs[1, k]), names.index(n))
        for k, n in enumerate(("right_shoulder_lift", "right_elbow_flex", "left_gripper"))
    ]
    for ax, j in panels:
        for k in range(100, len(demo), 100):
            ax.axvline(k / FPS, color=GRID, lw=0.8, zorder=0)
        ax.plot(t, demo[:, j], lw=1.6, color=C_DEMO, label="시연 (demo action)")
        ax.plot(t, pred[:, j], lw=1.3, color=C_MODEL, label="모델 (ACT, open-loop)")
        ax.set_title(names[j], loc="left", fontsize=12 if j == ji else 11, color=INK)
        ax.set_xlim(0, t[-1])
        style(ax, 9)
    ax, _ = panels[0]
    b = dm.baseline
    ax.axhline(b + cfg.sign * cfg.tilt_off, color=C_TILT, lw=1.1, ls=(0, (5, 3)))
    ax.axhline(b + cfg.sign * cfg.return_off, color=C_RET, lw=1.1, ls=(0, (5, 3)))
    ax.text(
        0.3,
        b + cfg.sign * cfg.tilt_off + 2,
        f"tilt +{cfg.tilt_off:.0f}°",
        color=INK2,
        fontsize=10,
        va="bottom",
    )
    ax.text(
        0.3,
        b + cfg.sign * cfg.return_off + 2,
        f"return +{cfg.return_off:.0f}°",
        color=INK2,
        fontsize=10,
        va="bottom",
    )
    for det, c in ((dd, C_DEMO), (dm, C_MODEL)):
        if det.state == DONE:
            ax.axvline(det.done_step / FPS, color=c, lw=1.2, ls=":")
    top = ax.get_ylim()[1]
    if dm.state == DONE:
        ax.text(
            dm.done_step / FPS + 0.3,
            top,
            f"DONE  모델 {dm.done_step / FPS:.1f} s / 시연 {dd.done_step / FPS:.1f} s",
            color=INK,
            fontsize=10,
            va="top",
        )
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK2, ncol=2, bbox_to_anchor=(0, 0.93))
    for ax, _ in panels[1:]:
        ax.set_xlabel("시간 (s)", color=INK2, fontsize=10)

    fig.suptitle(
        f"모델 출력 vs 시연: ep{episode} (12관절 MAE {row.mae_all:.2f}°, 100 에피소드 중앙값 수준)",
        x=0.012,
        y=0.975,
        ha="left",
        color=INK,
        fontsize=17,
    )
    fig.text(
        0.012,
        0.915,
        "open-loop, 학습 데이터: 매 청크 입력은 시연 영상·state (모델 출력이 다음 관측에 반영되지 않음), "
        "100 에피소드 모두 학습에 사용 → 일반화가 아닌 파이프라인 확인용. 회색 세로선 = 청크 경계(100스텝), 점선 = DONE.",
        color=INK2,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.85, bottom=0.08)
    save(fig, out / "model_vs_demo.png")


# ---------------------------------------------------------------- 시뮬 영상
def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def sim_events(run_dir: Path) -> dict[str, float]:
    """롤아웃 CSV → 장면 시각 (영상 s). 영상 프레임 = step + 1 (첫 장은 리셋)."""
    r = pd.read_csv(sorted(run_dir.glob("run_*.csv"))[0])
    lg, rg = r["obs_5"].values, r["obs_11"].values  # left/right gripper
    opened = np.flatnonzero((lg > 50) & (rg > 50))
    grasp = next(i for i in range(opened[0], len(r)) if lg[i] < 10 and rg[i] < 10)
    tilt = int(r.index[r.det.eq("POURING")][0])
    done = int(r.index[r.det.eq("DONE")][0])
    release = next(i for i in range(done, len(r)) if rg[i] > 50)
    f = lambda s: (s + 1) / FPS  # noqa: E731
    return dict(
        grasp=f(grasp + 15),
        tilt=f(tilt),
        peak=f(int(r.roll_obs.idxmax())),
        done=f(done),
        done_sub=f(done + 30),
        putdown=f(release + 10),
        end=f(len(r) - 1),
    )


def grab(video: Path, t: float) -> np.ndarray:
    raw = subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def fig_sim_stills(video: Path, ev: dict, out: Path):
    scenes = [
        ("grasp", "① 잡기", "왼손 컵 · 오른손 물통"),
        ("peak", "② 붓기 (최대 기울기)", "오른손목 roll 최대"),
        ("done_sub", "③ 완료 감지 + 안내", '"다 따랐습니다" 자막·음성'),
        ("putdown", "④ 내려놓기", f"DONE 후 {ev['putdown'] - ev['done']:.0f} s · 물통 놓기"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.85))
    for ax, (k, title, sub) in zip(axs, scenes):
        ax.imshow(grab(video, ev[k]))
        ax.set_axis_off()
        ax.text(0, 1.09, title, transform=ax.transAxes, fontsize=13, color=INK, va="bottom")
        ax.text(
            0,
            1.02,
            f"{sub} · {ev[k] - 1 / FPS:.1f} s",
            transform=ax.transAxes,
            fontsize=10,
            color=INK2,
            va="bottom",
        )
    fig.subplots_adjust(left=0.006, right=0.994, top=0.83, bottom=0.01, wspace=0.03)
    save(fig, out / "sim_stills.png")


def make_gif(video: Path, ev: dict, out: Path, width=480, fps=10):
    """느린 구간은 빨리 감고, 완료 직전~안내 자막 구간만 실제 속도."""
    a, b = ev["done"] - 2.0, ev["done"] + 4.5
    segs = [(0, ev["tilt"] - 1, 4), (ev["tilt"] - 1, a, 4), (a, b, 1), (b, ev["end"], 4)]
    parts = "".join(
        f"[0:v]trim={s:.2f}:{e:.2f},setpts=(PTS-STARTPTS)/{sp}[v{i}];" for i, (s, e, sp) in enumerate(segs)
    )
    graph = (
        parts + "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0,fps={fps},"
        f"scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    path = out / "demo_sim.gif"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-filter_complex",
            graph,
            "-loop",
            "0",
            str(path),
        ],
        check=True,
    )
    dur = sum((e - s) / sp for s, e, sp in segs)
    print(f"  {path.name:22s} {path.stat().st_size / 1e3:6.0f} KB  ({dur:.1f} s, {width}px, {fps} fps)")


def make_mp4(video: Path, out: Path):
    path = out / "demo_sim.mp4"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    print(f"  {path.name:22s} {path.stat().st_size / 1e3:6.0f} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=GITHUB / "docs" / "media")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--episode", type=int, default=None, help="model_vs_demo 에피소드 (기본: MAE 중앙값)")
    ap.add_argument("--sim-video", type=Path, default=SIM_VIDEO)
    ap.add_argument(
        "--sim-run", type=Path, default=SIM_RUN, help="그 영상을 찍은 pour_rollout 결과 폴더 (run_XX.csv)"
    )
    ap.add_argument("--skip-video", action="store_true", help="GIF/mp4 는 건너뜀 (stills 는 만듦)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    th = json.loads(THRESHOLDS.read_text())
    _, names, eps = load(args.dataset)
    ji = names.index(th["joint"])
    print(f"→ {args.out}")
    fig_detector_aligned(eps, ji, th, args.out)
    fig_joint_selection(eps, th["base_frames"], args.out)
    fig_model_vs_demo(th, args.out, args.episode)

    ev = sim_events(args.sim_run)
    fig_sim_stills(args.sim_video, ev, args.out)
    if not args.skip_video:
        make_gif(args.sim_video, ev, args.out)
        make_mp4(args.sim_video, args.out)


if __name__ == "__main__":
    main()
