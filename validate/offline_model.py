"""할 일 2: 오프라인 모델 검증 (open-loop).

데이터셋 프레임(카메라 3 + state)을 ACT 모델에 넣어 action 시퀀스를 만들고,
오른팔 wrist_roll 명령에 완료 감지기를 적용해 시연 궤적과 비교한다.

추론 경로는 lerobot-rollout 의 SyncInferenceEngine 과 같다
(prepare_observation_for_inference → preprocessor → policy → postprocessor).
ACT 는 n_action_steps(=100) 마다 청크를 새로 뽑으므로 그 프레임만 디코딩한다.

[주의] open-loop 다: 모델 출력과 무관하게 다음 입력은 시연 영상/state 이다.
       실제 로봇에서는 모델 출력이 다음 관측을 바꾸므로(closed-loop) 오차가 누적될 수 있다.
       또 100 에피소드 모두 학습에 쓴 데이터라 일반화 성능이 아니라 파이프라인 확인용이다.

실행:
  python offline_model.py                      # 전체 100 에피소드 (CPU 약 6분)
  python offline_model.py --episodes 0 1 2     # 일부
  python offline_model.py --stride 30          # 30 스텝마다 재추론 (청크 앞 30개만 사용)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASET, MODEL_REPO, OUTPUTS, THRESHOLDS  # noqa: E402
from pour_detector import DONE, PourDetector, PourDetectorConfig  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FPS = 30
CAMS = ("top", "left_wrist", "right_wrist")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_DEMO, C_MODEL, C_TILT, C_RET = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"


def load_policy(model: str, device: str):
    from huggingface_hub import snapshot_download

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    path = model if Path(model).is_dir() else snapshot_download(model)
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device, cfg.pretrained_path = device, path
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post, cfg


def obs_from_item(item) -> dict:
    """데이터셋 프레임 → 로봇 관측과 같은 형식 (state float32, 이미지 uint8 HWC)."""
    obs = {"observation.state": item["observation.state"].numpy()}
    for c in CAMS:
        obs[f"observation.images.{c}"] = item[f"observation.images.{c}"].permute(1, 2, 0).numpy()
    return obs


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--model", default=MODEL_REPO)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=None, help="재추론 간격 (기본: 모델 n_action_steps)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS,
                    help="완료 감지 임계값 (joint_analysis.py --install 로 갱신)")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "offline_model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.utils import prepare_observation_for_inference

    th = json.loads(args.thresholds.read_text())
    det_cfg = PourDetectorConfig(tilt_off=th["tilt_off"], return_off=th["return_off"], hold=th["hold"],
                                 base_frames=th["base_frames"], sign=th["sign"])

    ds = LeRobotDataset(args.dataset.name, root=args.dataset, return_uint8=True)
    names = ds.meta.features["action"]["names"]
    ji = names.index(th["joint"].replace("observation.state.", ""))
    policy, pre, post, cfg = load_policy(args.model, args.device)
    stride = args.stride or cfg.n_action_steps
    assert stride <= cfg.n_action_steps
    eps = args.episodes if args.episodes is not None else list(range(ds.meta.total_episodes))
    print(f"device={args.device} chunk={cfg.chunk_size} stride={stride} episodes={len(eps)} joint={names[ji]}")

    rows, store = [], {}
    t0 = time.time()
    for n_done, e in enumerate(eps):
        meta = ds.meta.episodes[e]
        lo, hi = meta["dataset_from_index"], meta["dataset_to_index"]
        L = hi - lo
        demo = np.stack([np.asarray(a) for a in ds.hf_dataset[lo:hi]["action"]]).astype(np.float32)
        pred = np.zeros_like(demo)
        jumps = []
        for k in range(0, L, stride):
            item = ds[lo + k]
            assert int(item["episode_index"]) == e and int(item["frame_index"]) == k
            with torch.inference_mode():
                b = prepare_observation_for_inference(obs_from_item(item), torch.device(args.device),
                                                      item["task"], ds.meta.robot_type)
                chunk = post(policy.predict_action_chunk(pre(b)))[0, :stride].numpy()
            n = min(stride, L - k)
            pred[k:k + n] = chunk[:n]
            if k > 0:
                jumps.append(np.abs(pred[k] - pred[k - 1]))
        store[e] = (demo, pred)

        dd = PourDetector(det_cfg).run(demo[:, ji])
        dm = PourDetector(det_cfg).run(pred[:, ji])
        jumps = np.stack(jumps) if jumps else np.zeros((1, demo.shape[1]))
        rows.append(dict(
            episode=e, len_s=L / FPS,
            mae_all=float(np.abs(pred - demo).mean()), mae_roll=float(np.abs(pred[:, ji] - demo[:, ji]).mean()),
            demo_done_t=dd.done_step / FPS if dd.state == DONE else None,
            model_detected=dm.state == DONE, model_final_state=dm.state,
            model_tilt_t=dm.tilt_step / FPS if dm.tilt_step is not None else None,
            model_done_t=dm.done_step / FPS if dm.state == DONE else None,
            model_roll_peak_dev=float(det_cfg.sign * (pred[:, ji] - dm.baseline).max()),
            chunk_jump_max=float(jumps.max()), chunk_jump_roll_max=float(jumps[:, ji].max()),
        ))
        el = time.time() - t0
        print(f"  ep{e:3d} {L:5d}f  MAE {rows[-1]['mae_all']:5.2f}  roll MAE {rows[-1]['mae_roll']:5.2f}  "
              f"감지 {dm.state:7s}  [{n_done + 1}/{len(eps)} {el:5.0f}s, 남은 ~{el / (n_done + 1) * (len(eps) - n_done - 1):4.0f}s]",
              flush=True)

    r = pd.DataFrame(rows)
    r["done_diff_s"] = r["model_done_t"] - r["demo_done_t"]
    r.to_csv(args.out / f"offline_stride{stride}.csv", index=False)
    np.savez_compressed(args.out / f"actions_stride{stride}.npz",
                        **{f"ep{e}_demo": d for e, (d, p) in store.items()},
                        **{f"ep{e}_model": p for e, (d, p) in store.items()}, names=np.array(names))

    d = r[r.model_detected]
    print(f"\n[결과 · open-loop · stride {stride}]")
    print(f"  모델 명령 기준 감지율: {len(d)}/{len(r)}")
    if len(r) - len(d):
        print("  미감지:", r[~r.model_detected][["episode", "model_final_state", "model_roll_peak_dev"]].to_dict("records"))
    print(f"  action MAE (12관절 평균): 중앙 {r.mae_all.median():.2f}, 최대 {r.mae_all.max():.2f}")
    print(f"  {names[ji]} MAE: 중앙 {r.mae_roll.median():.2f}, 최대 {r.mae_roll.max():.2f}")
    if len(d):
        q = d.done_diff_s.abs().quantile([0.5, 0.9, 1]).values
        print(f"  DONE 시점 차이 |모델-시연| (s): 중앙 {q[0]:.2f}, p90 {q[1]:.2f}, 최대 {q[2]:.2f}")
    print(f"  청크 경계 점프 (12관절 최대): 중앙 {r.chunk_jump_max.median():.1f}, 최대 {r.chunk_jump_max.max():.1f}")
    print(f"  청크 경계 점프 ({names[ji]}): 중앙 {r.chunk_jump_roll_max.median():.1f}, 최대 {r.chunk_jump_roll_max.max():.1f}")

    # ---- 그림 1: wrist_roll 시연 vs 모델, 에피소드별 ----
    nc = 10 if len(eps) > 10 else len(eps)
    nr = int(np.ceil(len(eps) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.6 * nr), squeeze=False)
    rr = r.set_index("episode")
    for ax, e in zip(axs.flat, eps):
        demo, pred = store[e]
        t = np.arange(len(demo)) / FPS
        base = PourDetector(det_cfg).run(pred[:, ji]).baseline
        ax.plot(t, demo[:, ji], lw=1.0, color=C_DEMO)
        ax.plot(t, pred[:, ji], lw=1.0, color=C_MODEL)
        ax.axhline(base + det_cfg.sign * det_cfg.tilt_off, color=C_TILT, lw=0.5, ls="--")
        ax.axhline(base + det_cfg.sign * det_cfg.return_off, color=C_RET, lw=0.5, ls="--")
        row = rr.loc[e]
        if row.model_detected:
            ax.axvline(row.model_done_t, color=INK, lw=0.8)
        ax.set_title(f"ep{e}" + ("" if row.model_detected else " 미감지"), fontsize=7, pad=2,
                     color=INK if row.model_detected else C_TILT)
        style(ax)
    for ax in list(axs.flat)[len(eps):]:
        ax.axis("off")
    fig.suptitle(f"{names[ji]} · 파랑 시연 action / 주황 모델 action (open-loop, stride {stride}) · 검정 세로선 모델 DONE",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out / f"wrist_roll_model_vs_demo_stride{stride}.png", dpi=90)
    plt.close(fig)

    # ---- 그림 2: MAE 가 가장 큰/중간 에피소드의 12관절 ----
    worst = int(r.loc[r.mae_all.idxmax(), "episode"])
    med = int(r.iloc[(r.mae_all - r.mae_all.median()).abs().argsort().iloc[0]]["episode"])
    for tag, e in (("median", med), ("worst", worst)):
        demo, pred = store[e]
        t = np.arange(len(demo)) / FPS
        fig, axs = plt.subplots(4, 3, figsize=(15, 9), sharex=True)
        for j, ax in enumerate(axs.T.flat):
            ax.plot(t, demo[:, j], lw=1.0, color=C_DEMO)
            ax.plot(t, pred[:, j], lw=1.0, color=C_MODEL)
            for k in range(stride, len(demo), stride):
                ax.axvline(k / FPS, color=GRID, lw=0.6)
            ax.set_title(names[j], fontsize=9, loc="left", color=INK)
            style(ax)
        fig.suptitle(f"ep{e} ({tag} MAE {rr.loc[e].mae_all:.2f}) · 파랑 시연 / 주황 모델 · 회색 세로선 = 청크 경계",
                     x=0.01, ha="left", color=INK)
        fig.tight_layout()
        fig.savefig(args.out / f"joints_ep{e}_{tag}_stride{stride}.png", dpi=90)
        plt.close(fig)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
