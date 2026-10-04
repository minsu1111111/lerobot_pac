"""에피소드 관절 궤적을 양팔 SO-101 MuJoCo 로 재생 → mp4 / 인터랙티브 창.

  # 데이터셋 action (또는 observation.state) 재생 (LeRobot v3 데이터셋 폴더, --dataset 으로 지정)
  python play_trajectory.py --source dataset --episode 0 --field action --stills auto
  # 모델 출력 npz (keys ep{N}_demo, ep{N}_model, names). both = 모델(불투명) + 데모(반투명 분홍 겹침)
  python play_trajectory.py --source npz --file actions.npz --episode 0 --which both
  # 물리 모드 (위치 액추에이터 추종) + 추종오차 출력, 창으로 보기
  python play_trajectory.py --episode 0 --physics --window

출력: <outputs>/ep{N}_{이름}.mp4, --stills 면 같은 이름_f{프레임}.png
  <outputs> = --out-dir, 없으면 sim_paths.outputs_dir() (SIM_OUTPUTS → $UNITA_LOCAL/outputs/sim → ../../local/outputs/sim → sim/outputs)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_bi_so101 import GRIP_CLOSE, GRIP_OPEN, KP_DEFAULT, NAMES, MujocoBiSO101, lerobot_to_qpos, qpos_to_lerobot  # noqa: E402
from sim_paths import local_root, outputs_dir  # noqa: E402

DATASET_DEFAULT = Path.home() / ".cache/huggingface/lerobot/UNITAmanipulation/bi_so101_pour_water_20260920_194823"
_LR = local_root()
NPZ_DEFAULT = _LR / "outputs" / "offline_model" / "actions_stride100.npz" if _LR else None  # UNITA offline_check.py 출력
OUT_DIR = outputs_dir()
FPS = 30
RWR = NAMES.index("right_wrist_roll")


def load_dataset(ep: int, field: str, dataset: Path = DATASET_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """(궤적 T×12, 같은 에피소드 observation.state T×12). LeRobot v3 parquet (data/chunk-*/file-*.parquet)."""
    import pandas as pd

    files = sorted(glob.glob(str(Path(dataset) / "data/chunk-*/file-*.parquet")))
    if not files:
        raise SystemExit(f"데이터셋 parquet 없음: {dataset} (--dataset 으로 지정)")
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"])
        df = df[df.episode_index == ep]
        if len(df):
            df = df.sort_values("frame_index")
            return np.stack(df[field].values), np.stack(df["observation.state"].values)
    raise SystemExit(f"에피소드 {ep} 없음")


def load_npz(path: Path, ep: int) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    if "names" in z:  # 순서 확인
        names = [str(n).removesuffix(".pos") for n in z["names"]]
        assert names == NAMES, f"관절 순서 다름: {names}"
    out = {k: z[f"ep{ep}_{k}"] for k in ("demo", "model") if f"ep{ep}_{k}" in z}
    if not out:
        eps = sorted({k.split("_")[0] for k in z.files if k.startswith("ep")})
        raise SystemExit(f"{path} 에 ep{ep} 없음 (있는 것: {eps})")
    return out


def key_frames(traj: np.ndarray) -> dict[str, int]:
    """대표 장면 프레임: 시작, 컵 잡기(왼 그리퍼 열린 뒤 닫힘), 붓기 최대(오른 wrist_roll 최대), 복귀 후."""
    g = traj[:, NAMES.index("left_gripper")]
    opened = np.flatnonzero(g > GRIP_OPEN)
    grasp = next((i for i in range(opened[0], len(g)) if g[i] < GRIP_CLOSE), None) if len(opened) else None
    r = traj[:, RWR]
    peak = int(r.argmax())
    base = np.median(r[:30])
    back = next((i for i in range(peak, len(r)) if r[i] - base < 0.3 * (r[peak] - base)), len(r) - 1)
    kf = {"rest": 0, "cup_grasp": grasp, "pour_peak": peak, "after_return": min(back + 30, len(r) - 1)}
    return {k: int(v) for k, v in kf.items() if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dataset", "npz"], default="dataset")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--dataset", type=Path, default=DATASET_DEFAULT, help="LeRobot 데이터셋 폴더 (source=dataset)")
    ap.add_argument("--field", choices=["action", "observation.state"], default="action")
    ap.add_argument("--file", type=Path, default=NPZ_DEFAULT, help="모델 출력 npz (source=npz)")
    ap.add_argument("--which", choices=["demo", "model", "both"], default="both")
    ap.add_argument("--physics", action="store_true", help="위치 액추에이터 물리 추종 (기본: 운동학적 재생)")
    ap.add_argument("--window", action="store_true", help="mujoco.viewer 창으로 실시간 재생")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--camera", default="front", help="front | top | side")
    ap.add_argument("--spacing", type=float, default=0.45, help="두 팔 받침 간격 (m)")
    ap.add_argument("--yaw", type=float, default=30.0, help="두 팔 안쪽 회전 (deg)")
    ap.add_argument("--no-props", action="store_true", help="컵·물통 모형 끄기")
    ap.add_argument("--stills", default="", help="'auto' 또는 프레임 번호 목록 '0,360,1317'")
    ap.add_argument("--kp", type=float, default=KP_DEFAULT, help="물리 모드 액추에이터 kp (998.22 = MJCF 원래 값)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="mp4 경로 (기본: <out-dir>/<tag>.mp4)")
    ap.add_argument("--out-dir", type=Path, default=None, help=f"결과 폴더 (기본: {OUT_DIR})")
    a = ap.parse_args()
    out_dir = a.out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ghost = None
    obs = None
    if a.source == "dataset":
        traj, obs = load_dataset(a.episode, a.field, a.dataset)
        tag = f"ep{a.episode}_{a.field.replace('observation.', '')}"
        label = a.field
    else:
        if a.file is None:
            raise SystemExit("--file 로 npz 경로를 지정하세요")
        z = load_npz(a.file, a.episode)
        if a.which == "both":
            assert {"demo", "model"} <= set(z), f"demo/model 둘 다 필요 (있는 것: {list(z)})"
            n = min(len(z["demo"]), len(z["model"]))
            traj, ghost = z["model"][:n], z["demo"][:n]
            label = "model (solid) / demo (pink ghost)"
        else:
            traj = z[a.which]
            label = a.which
        tag = f"ep{a.episode}_npz_{a.which}"
    tag += ("_phys" + (f"_kp{a.kp:g}" if a.kp != KP_DEFAULT else "")) if a.physics else ""
    tag += f"_{a.camera}" if a.camera != "front" else ""
    sl = slice(a.start, a.end)
    traj = traj[sl]
    ghost = ghost[sl] if ghost is not None else None
    obs = obs[sl] if obs is not None else None

    video = None if a.no_video else (a.out or out_dir / f"{tag}.mp4")
    stills = {}
    if a.stills == "auto":
        stills = {v: k for k, v in key_frames(traj).items()}
    elif a.stills:
        stills = {int(s): "" for s in a.stills.split(",")}

    sim = MujocoBiSO101(render="window" if a.window else ("offscreen" if (video or stills) else "none"),
                        video_path=video, fps=FPS, physics=a.physics, spacing=a.spacing,
                        ghost=ghost is not None, props=not a.no_props, camera=a.camera, kp=a.kp, yaw_deg=a.yaw)
    sim.place_props_from_trajectory(traj)
    if ghost is not None:
        sim.set_ghost(ghost[0])
    def save_still(t):
        if t in stills and sim.renderer is not None:
            from PIL import Image

            p = out_dir / f"{tag}{'_' + stills[t] if stills[t] else ''}_f{a.start + t}.png"
            Image.fromarray(sim.render_frame()).save(p)
            print("still", p)

    def caption(t):
        txt = f"ep{a.episode} {label}  t={(a.start + t) / FPS:5.1f}s  f={a.start + t}\nR wrist_roll cmd {traj[t, RWR]:6.1f}"
        return txt + (f"  demo {ghost[t, RWR]:6.1f}" if ghost is not None else "")

    sim.text = caption(0)
    sim_states = [sim.reset(obs[0] if obs is not None else traj[0])]
    save_still(0)
    for t in range(1, len(traj)):
        if ghost is not None:
            sim.set_ghost(ghost[t])
        sim.text = caption(t)
        sim_states.append(sim.step(traj[t]))
        save_still(t)
        if not sim.is_running():
            break
    sim.close()
    if video:
        print("video", video)

    if a.physics:  # 추종오차: 시뮬 상태(t) vs 명령(t) / vs 실제 기록 상태(t)
        S = np.array(sim_states)
        n = len(S)
        cmd = np.array([qpos_to_lerobot(lerobot_to_qpos(x, sim.m)) for x in traj[:n]])  # MJCF 범위로 잘린 명령
        rep = {"cmd": np.abs(S[1:] - cmd[1:]), "clip": np.abs(cmd[1:] - traj[1:n])}
        if obs is not None:
            rep["real_state"] = np.abs(S[1:] - obs[1:n])
            rep["real_state_vs_cmd"] = np.abs(obs[1:n] - traj[1:n])
        summary = {}
        print(f"\n추종오차 (deg, gripper 는 0-100) — {n} 프레임")
        print(f"{'joint':20s}" + "".join(f"{k + ' mean/p95/max':>31s}" for k in rep))
        for j, name in enumerate(NAMES):
            row = {k: [float(v[:, j].mean()), float(np.percentile(v[:, j], 95)), float(v[:, j].max())] for k, v in rep.items()}
            summary[name] = row
            print(f"{name:20s}" + "".join(f"{r[0]:10.2f}{r[1]:10.2f}{r[2]:10.2f}" for r in row.values()))
        jp = out_dir / f"{tag}_tracking.json"
        jp.write_text(json.dumps(summary, indent=1))
        print("tracking", jp)


if __name__ == "__main__":
    main()
