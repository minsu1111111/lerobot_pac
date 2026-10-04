"""현장(인터넷 없음) 점검: 데모에 필요한 것이 전부 로컬에서 로딩되는지 확인한다.

  python offline_check.py                    # 전체
  python offline_check.py --quick            # run_demo.sh 용: Whisper 로딩·ResNet 실험·데이터셋 생략
  python offline_check.py --block-network    # 소켓 연결 자체를 막아 진짜 오프라인 흉내

항목: 파이썬/conda env, torch/CUDA/VRAM, lerobot, 정책 체크포인트 로딩 + 더미 추론(시간, 최대 VRAM),
     ResNet18 ImageNet 가중치 위험(아래), 데이터셋 메타(선택), TTS wav, Whisper small.pt, 오디오 장치.

[ResNet18 위험] ACT config 의 pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1" 이라
  정책을 '생성'할 때 torchvision 이 ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth 를 찾고,
  없으면 인터넷에서 받으려 한다 (체크포인트를 덮어쓰기 전 단계). 이 검사는 빈 TORCH_HOME + 네트워크 차단
  하위 프로세스로 (a) 원래 설정 (b) backbone 가중치 None 두 경우를 실제로 생성해 보고,
  (c) 두 방식의 추론 출력이 같은지(= None 으로 꺼도 안전한지) 비교한다.
  pour_rollout.py 는 기본으로 None 을 쓴다.

종료 코드: FAIL 이 하나라도 있으면 1.
"""

import os

# import 전에 오프라인 모드 (huggingface_hub / transformers / datasets)
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

GITHUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GITHUB))
sys.path.insert(0, str(GITHUB / "rollout"))
from paths import DATASET, MODEL_REPO  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []  # (항목, PASS/WARN/FAIL/SKIP, 설명)


def record(name, status, msg=""):
    RESULTS.append((name, status, msg))
    print(f"  [{status:4s}] {name}: {msg}", flush=True)


def check(name, fail_status="FAIL"):
    """데코레이터: 예외가 나면 fail_status 로 기록."""

    def deco(fn):
        def run(*a, **kw):
            t = time.perf_counter()
            try:
                out = fn(*a, **kw)
                status, msg = out if isinstance(out, tuple) else ("PASS", str(out or ""))
            except Exception as e:
                status, msg = (
                    fail_status,
                    f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}",
                )
                if os.environ.get("OFFLINE_CHECK_DEBUG"):
                    traceback.print_exc()
            record(name, status, f"{msg} ({time.perf_counter() - t:.1f}s)")
            return status

        return run

    return deco


def block_network():
    """외부 연결 차단: socket.create_connection 과 AF_INET(6) connect 를 막는다 (유닉스 소켓은 허용)."""

    def deny(*a, **kw):
        raise OSError("offline_check: network blocked")

    socket.create_connection = deny
    orig = socket.socket.connect

    def connect(self, addr):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError(f"offline_check: network blocked ({addr})")
        return orig(self, addr)

    socket.socket.connect = connect


# --------------------------------------------------------------------------- #
@check("python / conda env")
def c_env():
    exe = sys.executable
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    ok = "lerobot" in exe or env == "lerobot"
    msg = f"{exe} (py {sys.version.split()[0]}, CONDA_DEFAULT_ENV={env or '-'})"
    return ("PASS" if ok else "WARN"), msg


@check("torch / CUDA")
def c_torch():
    import torch

    msg = f"torch {torch.__version__}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "PASS", f"{msg}, {p.name}, VRAM {p.total_memory / 2**30:.1f}GB, CUDA {torch.version.cuda}"
    return "WARN", f"{msg}, CUDA 없음 → CPU 추론 (느림, 30fps 루프에서 청크마다 지연)"


@check("lerobot import")
def c_lerobot():
    import lerobot
    import lerobot.policies  # noqa: F401
    import lerobot.robots.bi_so_follower  # noqa: F401
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401

    return f"v{getattr(lerobot, '__version__', '?')} ← {Path(lerobot.__file__).parent}"


@check("정책 로딩 + 더미 추론")
def c_policy(args):
    import numpy as np
    import torch
    from pour_rollout import Policy

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    pol = Policy(args.policy, dev, None, None)
    load_s = time.perf_counter() - t
    ts = pol.warmup(n=3)  # zeros 관측 (state 12, top 480x640, wrist 240x320 x2)
    a = pol(
        {
            **dict.fromkeys(__import__("backends").JOINT_KEYS, 0.0),
            **{c: np.zeros(s, np.uint8) for c, s in __import__("backends").CAM_SHAPES.items()},
        }
    )
    assert a.shape == (12,) and np.isfinite(a).all()
    msg = (
        f"{dev}, 로딩 {load_s:.1f}s, 청크 추론 첫 {ts[0] * 1e3:.0f}ms / 이후 {min(ts[1:]) * 1e3:.0f}ms, "
        f"chunk={pol.cfg.chunk_size} n_action_steps={pol.cfg.n_action_steps} ← {pol.path}"
    )
    if dev == "cuda":
        msg += f", 최대 VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f}MB"
    return msg


@check("ResNet18 ImageNet 캐시 파일", fail_status="WARN")
def c_resnet_file():
    import torch

    d = Path(torch.hub.get_dir()) / "checkpoints"
    f = sorted(d.glob("resnet18-*.pth"))
    if f:
        return "PASS", f"{f[0]} ({f[0].stat().st_size / 2**20:.0f}MB) — 원래 설정으로 생성해도 다운로드 안 함"
    return "WARN", f"{d} 에 없음 → pretrained_backbone_weights 를 끄지 않으면 정책 생성 때 다운로드 시도"


def probe_backbone(mode: str, policy: str):
    """하위 프로세스용: 빈 TORCH_HOME + 네트워크 차단 상태에서 정책 생성."""
    block_network()
    import numpy as np
    import torch
    from pour_rollout import Policy

    torch.manual_seed(0)
    pol = Policy(policy, "cpu", None, None, keep_backbone_download=(mode == "default"))
    from backends import CAM_SHAPES, JOINT_KEYS

    rng = np.random.default_rng(0)
    obs = {k: float(v) for k, v in zip(JOINT_KEYS, rng.normal(0, 30, 12))}
    obs.update({c: rng.integers(0, 255, s, dtype=np.uint8) for c, s in CAM_SHAPES.items()})
    a = pol(obs)
    print("PROBE_OK " + json.dumps(a.tolist()))


@check("ResNet18 오프라인 위험 실험", fail_status="FAIL")
def c_resnet_probe(args):
    """빈 TORCH_HOME 에서 (a) 원래 설정 (b) None 생성. (c) 진짜 캐시로 원래 설정 출력과 None 출력 비교."""
    import numpy as np

    def run(mode, torch_home):
        env = dict(os.environ, TORCH_HOME=torch_home, CUDA_VISIBLE_DEVICES="")
        p = subprocess.run(
            [sys.executable, __file__, "--_probe", mode, "--policy", args.policy],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_OK ")]
        if line:
            return np.array(json.loads(line[0][9:])), ""
        err = (p.stderr.strip().splitlines() or ["?"])[-1]
        return None, err

    with tempfile.TemporaryDirectory() as empty:
        a_def, err_def = run("default", empty)
        a_none, err_none = run("none", empty)
    real_home = os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch"))
    a_def_cached, err_c = run("default", real_home)
    parts = [
        f"(a) 캐시 없음+원래 설정: {'생성됨' if a_def is not None else '실패 → ' + err_def[:120]}",
        f"(b) 캐시 없음+None: {'생성됨' if a_none is not None else '실패 → ' + err_none[:120]}",
    ]
    if a_def_cached is not None and a_none is not None:
        diff = float(np.abs(a_def_cached - a_none).max())
        parts.append(f"(c) 원래(캐시 사용) vs None 출력 최대차 {diff:.2e}")
        same = diff < 1e-4
    else:
        parts.append(f"(c) 비교 불가 ({err_c[:80]})")
        same = False
    status = "PASS" if (a_none is not None and same) else "FAIL"
    return status, " | ".join(parts)


@check("데이터셋 메타 (선택)", fail_status="WARN")
def c_dataset():
    if not DATASET.exists():
        return "SKIP", f"{DATASET} 없음 (데모에는 불필요, replay 시험에만 필요)"
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    m = LeRobotDatasetMetadata(DATASET.name, root=DATASET)
    return f"{m.total_episodes} 에피소드, {m.total_frames} 프레임, robot_type={m.robot_type}"


@check("TTS wav")
def c_tts():
    sys.path.insert(0, str(GITHUB / "tts"))
    from phrases import PHRASES

    d = GITHUB / "tts" / "wavs"
    missing = [k for k in PHRASES if not (d / f"{k}.wav").is_file()]
    if missing:
        return "FAIL", f"{d} 에 없음: {missing}"
    players = [b for b in ("aplay", "paplay") if shutil.which(b)]
    return ("PASS" if players else "WARN"), f"{len(PHRASES)}개 모두 있음, 재생기 {players or '없음(자막만)'}"


@check("Whisper small.pt", fail_status="FAIL")
def c_whisper(load: bool):
    f = Path.home() / ".cache/whisper/small.pt"
    if not f.is_file():
        return "FAIL", f"{f} 없음 (인터넷 되는 곳에서 whisper.load_model('small') 한 번 실행)"
    msg = f"{f} ({f.stat().st_size / 2**20:.0f}MB)"
    if not load:
        return "PASS", msg + ", 로딩 생략"
    import whisper

    t = time.perf_counter()
    whisper.load_model("small", device="cpu")
    return "PASS", msg + f", CPU 로딩 {time.perf_counter() - t:.1f}s"


@check("오디오 장치", fail_status="WARN")
def c_audio():
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        ins = [d["name"] for d in devs if d["max_input_channels"] > 0]
        outs = [d["name"] for d in devs if d["max_output_channels"] > 0]
        return (
            "PASS" if ins and outs else "WARN"
        ), f"입력 {len(ins)}개 {ins[:3]}, 출력 {len(outs)}개 {outs[:3]}"
    except Exception as e:  # PortAudio 없음 등 → ALSA 목록으로 대체
        msg = f"sounddevice 불가({type(e).__name__}: {e}); "
        out = []
        for cmd in (["aplay", "-l"], ["arecord", "-l"]):
            if shutil.which(cmd[0]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                cards = [ln for ln in r.stdout.splitlines() if ln.startswith("card")]
                out.append(f"{cmd[0]}: {len(cards)}개 {cards[:2]}")
        return "WARN", msg + " | ".join(out) + " → STT 마이크는 PortAudio(libportaudio2) 필요"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Whisper 로딩·ResNet 실험·데이터셋 생략")
    ap.add_argument("--skip-whisper", action="store_true", help="Whisper 로딩 생략 (파일 존재만 확인)")
    ap.add_argument("--block-network", action="store_true", help="소켓 연결 차단")
    ap.add_argument("--no-stt", action="store_true", help="STT 안 씀 → Whisper·마이크 검사 SKIP")
    ap.add_argument("--policy", default=MODEL_REPO)
    ap.add_argument("--_probe", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._probe:
        probe_backbone(args._probe, args.policy)
        return 0

    if args.block_network:
        block_network()
    print(
        f"[offline_check] HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1"
        f"{' + 네트워크 차단' if args.block_network else ''}"
    )
    c_env()
    c_torch()
    c_lerobot()
    c_policy(args)
    c_resnet_file()
    if args.quick:
        record("ResNet18 오프라인 위험 실험", "SKIP", "--quick")
        record("데이터셋 메타 (선택)", "SKIP", "--quick")
    else:
        c_resnet_probe(args)
        c_dataset()
    c_tts()
    if args.no_stt:
        record("Whisper small.pt", "SKIP", "--no-stt")
    else:
        c_whisper(load=not (args.quick or args.skip_whisper))
    c_audio()

    w = max(len(n) for n, _, _ in RESULTS)
    print("\n" + "=" * 72)
    for n, s, m in RESULTS:
        print(f"{s:4s}  {n:<{w}}  {m[:150]}")
    print("=" * 72)
    fails = [n for n, s, _ in RESULTS if s == "FAIL"]
    print(
        f"결과: {'FAIL ' + str(fails) if fails else 'PASS'} "
        f"(WARN {sum(s == 'WARN' for _, s, _ in RESULTS)}개)"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
