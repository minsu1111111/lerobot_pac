"""카메라·그리퍼 마운트 점검 도구 (손목 카메라 수위 시험 포함). 절차는 CAMERA_CHECK.md.

  PY=~/miniconda3/envs/lerobot-gpu/bin/python   # 없으면 envs/lerobot
  $PY tools/camera_check.py list                                   # 장치 목록 + 640x480/320x240@30 열리는지
  $PY tools/camera_check.py check --cam CAM --width 320 --height 240   # fps·드랍·밝기·포화·선명도 PASS/WARN
  $PY tools/camera_check.py focus --cam CAM --width 320 --height 240   # 렌즈 돌리며 선명도 실시간 (q 종료)
  $PY tools/camera_check.py exposure --cam CAM --exposure 150 --wb-temp 4600   # 수동 노출/WB + 전후 PNG
  $PY tools/camera_check.py capture --cam CAM --label level_target --count 3
  $PY tools/camera_check.py compare --labels level_minus1cm level_target level_plus1cm

CAM 은 /dev/v4l/by-id/...-video-index0 같은 고정 경로 권장 (/dev/videoN 은 재연결마다 바뀜).
결과물: <OUTPUTS>/camera_check/<YYYY-MM-DD>/  (OUTPUTS = paths.py, 없으면 ./outputs)

주의
- 카메라는 한 프로세스만 연다. lerobot-teleoperate/record, 다른 camera_check 가 켜져 있으면 실패한다.
- v4l2-ctl 로 바꾼 노출/WB 는 USB 재연결·재부팅 때 초기화된다 (exposure 가 매번 쓸 .sh 를 저장해 줌).
- 선명도(Laplacian 분산)는 장면 의존이다: 같은 무늬 있는 대상(인쇄 글씨)에서, 같은 해상도로만 비교할 것.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 최상위
    from paths import OUTPUTS  # noqa: E402
except Exception:  # paths.py 없이 단독 복사해 쓴 경우
    OUTPUTS = Path.cwd() / "outputs"

OUT_ROOT = OUTPUTS / "camera_check"
MODES = [(640, 480, 30), (320, 240, 30)]  # 학습 데이터: top 640x480, 손목 320x240, 30fps
FPS_MIN = 28.0
SAT_MAX_PCT = 2.0
# 선명도 절대 하한. 지난 프로젝트(320x240): 초점 나간 손목캠 중앙값 9.5, 정상 씬캠 147.
SHARP_FLOOR = 30.0
SHARP_REL = 0.7  # focus 에서 찍은 최고값 대비 이 비율 이상이면 PASS
BUSY_HINT = (
    "  - 다른 프로그램이 카메라를 쓰고 있지 않은지 확인 (lerobot-teleoperate/record, 다른 camera_check, 브라우저, cheese 등)\n"
    "  - 경로가 맞는지: python camera_check.py list  /  ls -l /dev/v4l/by-id/\n"
    "  - video-index1 은 메타데이터 노드라 영상이 안 나옴 → video-index0 을 쓸 것\n"
    "  - USB 를 다시 꽂았다면 번호가 바뀌었을 수 있음 (by-id 경로 권장)"
)


# ----------------------------------------------------------------------------- 공용


def say(msg: str = "") -> None:
    print(msg, flush=True)


def day_dir(day: str | None = None) -> Path:
    d = OUT_ROOT / (day or date.today().isoformat())
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug_of(cam: str) -> str:
    """'/dev/v4l/by-id/usb-XYZ_Camera-video-index0' → 'XYZ_Camera' (파일 이름용)."""
    s = os.path.basename(str(cam))
    s = re.sub(r"^(usb-|pci-)", "", s)
    s = re.sub(r"-video-index\d+$", "", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:48] or "cam"


def fourcc_str(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    s = "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))
    return s if s.isprintable() and s.strip() else "?"


def users_of(cam: str) -> list[str]:
    """이 장치를 열고 있는 프로세스 (권한 되는 범위: 보통 내 프로세스)."""
    try:
        node = os.path.realpath(cam)
    except OSError:
        return []
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                if os.path.realpath(os.path.join(fd_dir, fd)) == node:
                    cmd = " ".join(Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split())
                    found.append(f"pid {pid}: {cmd.strip()[:120]}")
                    break
        except OSError:
            continue
    return found


def explain_open_failure(cam: str, what: str) -> None:
    say(f"[오류] 카메라 {what}: {cam}")
    if not str(cam).isdigit():
        if not os.path.exists(cam):
            say("  - 경로가 없음 (카메라가 안 꽂혔거나 경로 오타)")
        elif (
            Path(f"/sys/class/video4linux/{os.path.basename(os.path.realpath(cam))}/index").exists()
            and Path(f"/sys/class/video4linux/{os.path.basename(os.path.realpath(cam))}/index")
            .read_text()
            .strip()
            != "0"
        ):
            say(
                f"  - {os.path.realpath(cam)} 는 메타데이터 노드(index≠0)라 영상이 안 나옴 → 같은 카메라의 video-index0 을 쓸 것"
            )
        elif not os.access(os.path.realpath(cam), os.R_OK | os.W_OK):
            say("  - 권한 없음: sudo usermod -aG video $USER 후 재로그인")
        users = users_of(cam)
        if users:
            say("  - 지금 이 카메라를 쓰는 프로세스:")
            for u in users:
                say(f"      {u}")
    say(BUSY_HINT)


def open_cam(cam: str, width: int | None, height: int | None, fps: float | None, fourcc: str | None):
    """V4L2 로 연다. FOURCC → 크기 → fps 순서로 설정 (lerobot OpenCVCamera 와 같은 순서).

    실패하면 이유를 출력하고 None. 성공하면 (cap, 첫 프레임).
    """
    src = int(cam) if str(cam).isdigit() else str(cam)
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        explain_open_failure(cam, "열기 실패")
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].ljust(4)))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    frame = None
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
    if frame is None:
        cap.release()
        explain_open_failure(cam, "열렸지만 프레임이 안 나옴")
        return None
    return cap, frame


def warmup(cap: cv2.VideoCapture, seconds: float = 1.0):
    """자동 노출/WB 가 자리 잡게 프레임을 버린다. 마지막 프레임 반환."""
    frame, t0 = None, time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


def center_box(shape, frac: float):
    h, w = shape[:2]
    f = max(0.05, min(1.0, frac))
    cw, ch = int(w * f), int(h * f)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, cw, ch


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def metrics(frame: np.ndarray, roi: float = 0.4) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x0, y0, cw, ch = center_box(frame.shape, roi)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "brightness": float(gray.mean()),
        "sat_pct": float((frame.max(axis=2) >= 250).mean() * 100),  # 어느 채널이든 하얗게 날아간 픽셀
        "dark_pct": float((gray <= 5).mean() * 100),
        "colorfulness": float(hsv[..., 1].mean()),  # HSV 채도 평균 (자동 노출에 색이 날아가면 낮아짐)
        "sharp": sharpness(gray),
        "sharp_center": sharpness(gray[y0 : y0 + ch, x0 : x0 + cw]),
    }


def fmt_metrics(m: dict) -> str:
    return (
        f"밝기 {m['brightness']:.0f}/255, 포화 {m['sat_pct']:.2f}%, 암부 {m['dark_pct']:.1f}%, "
        f"채도 {m['colorfulness']:.0f}, 선명도 전체 {m['sharp']:.0f} / 중앙 {m['sharp_center']:.0f}"
    )


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def tile(items: list[tuple[str, np.ndarray]], height: int, sub: list[str] | None = None) -> np.ndarray:
    """이미지를 같은 높이로 맞춰 가로로 붙이고 위에 이름표 (OpenCV 글꼴이라 영문만)."""
    tiles = []
    for i, (title, img) in enumerate(items):
        s = height / img.shape[0]
        t = cv2.resize(img, (int(round(img.shape[1] * s)), height), interpolation=cv2.INTER_NEAREST)
        bar_h = 44 if sub else 26
        bar = np.zeros((bar_h, t.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if sub:
            cv2.putText(bar, sub[i], (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        t = np.vstack([bar, t])
        tiles.append(np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255))  # 흰 구분선
    return np.hstack(tiles)[:, :-4]


def stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        out.append(np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=255))
    return np.vstack(out)[:-6]


# ----------------------------------------------------------------------------- v4l2-ctl

ALIASES = {
    "auto_exposure": ["auto_exposure", "exposure_auto"],
    "exposure": ["exposure_time_absolute", "exposure_absolute"],
    "wb_auto": ["white_balance_automatic", "white_balance_temperature_auto"],
    "wb_temp": ["white_balance_temperature"],
    "dyn_fps": ["exposure_dynamic_framerate", "exposure_auto_priority"],
}


def v4l2(*args: str, timeout: float = 5.0) -> tuple[int, str]:
    if not shutil.which("v4l2-ctl"):
        return 127, "v4l2-ctl 없음 (sudo apt install v4l-utils)"
    try:
        r = subprocess.run(["v4l2-ctl", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return 1, str(e)


def read_ctrls(cam: str) -> dict[str, dict]:
    """--list-ctrls-menus → {이름: {type, min, max, default, value, flags, menu{번호:이름}}}"""
    rc, out = v4l2("-d", cam, "--list-ctrls-menus")
    ctrls: dict[str, dict] = {}
    if rc != 0:
        return ctrls
    last = None
    for line in out.splitlines():
        m = re.match(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$", line)
        if m:
            info = {"type": m.group(2), "menu": {}}
            for k, v in re.findall(r"(\w+)=(\S+)", m.group(3)):
                info[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
            ctrls[m.group(1)] = last = info
            continue
        mm = re.match(r"^\s+(\d+):\s*(.+)$", line)
        if mm and last is not None:
            last["menu"][int(mm.group(1))] = mm.group(2).strip()
    return ctrls


def pick(ctrls: dict, key: str) -> str | None:
    return next((n for n in ALIASES[key] if n in ctrls), None)


def ctrl_summary(cam: str) -> str | None:
    c = read_ctrls(cam)
    if not c:
        return None
    parts = []
    for key in ("auto_exposure", "exposure", "wb_auto", "wb_temp", "dyn_fps"):
        n = pick(c, key)
        if n:
            v = c[n].get("value")
            menu = c[n]["menu"].get(v) if isinstance(v, int) else None
            parts.append(f"{n}={v}" + (f"({menu})" if menu else ""))
    return ", ".join(parts)


# ----------------------------------------------------------------------------- list


def cmd_list(a) -> int:
    links: dict[str, list[str]] = {}
    for d in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        if os.path.isdir(d):
            for n in sorted(os.listdir(d)):
                p = os.path.join(d, n)
                links.setdefault(os.path.realpath(p), []).append(p)
    nodes = sorted(
        (f"/dev/{n}" for n in os.listdir("/dev") if re.fullmatch(r"video\d+", n)),
        key=lambda s: int(re.sub(r"\D", "", s)),
    )
    if not nodes:
        say("[오류] /dev/video* 가 없음. 카메라 USB 연결을 확인 (lsusb 에 보이는지).")
        return 1
    has_v4l2 = bool(shutil.which("v4l2-ctl"))
    if not has_v4l2:
        say("(v4l2-ctl 없음: 형식·컨트롤 목록과 exposure 명령은 못 씀. sudo apt install v4l-utils)")
    say(
        f"OpenCV {cv2.__version__}, 시험 모드: "
        + ", ".join(f"{w}x{h}@{f}" for w, h, f in MODES)
        + (f", FOURCC={a.fourcc}" if a.fourcc else ", FOURCC=자동")
    )
    bad = 0
    for node in nodes:
        sysd = f"/sys/class/video4linux/{os.path.basename(node)}"
        name = Path(sysd, "name").read_text().strip() if os.path.exists(f"{sysd}/name") else "?"
        idx = Path(sysd, "index").read_text().strip() if os.path.exists(f"{sysd}/index") else "?"
        capture = idx == "0"
        if has_v4l2:
            rc, info = v4l2("-d", node, "--info")
            m = re.search(r"Device Caps\s*:\s*\S+\n((?:\s{2,}.+\n?)+)", info)
            if m:
                capture = "Video Capture" in m.group(1)
        say("")
        say(f"== {node}  [{name}]  index{idx} → {'영상(capture)' if capture else '메타데이터 (쓰지 말 것)'}")
        for p in links.get(node, []):
            say(f"   {p}")
        if not capture:
            continue
        users = users_of(node)
        if users:
            say("   [사용 중] " + " | ".join(users))
        for w, h, f in MODES:
            r = open_cam(node, w, h, f, a.fourcc) if not users else None
            if r is None:
                say(f"   {w}x{h}@{f}: 열기 실패")
                bad += 1
                continue
            cap, frame = r
            aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = cap.get(cv2.CAP_PROP_FPS)
            ok = (aw, ah) == (w, h) and frame.shape[1::-1] == (w, h) and afps >= FPS_MIN
            say(
                f"   {w}x{h}@{f}: 실제 {aw}x{ah} (프레임 {frame.shape[1]}x{frame.shape[0]}) @ {afps:.1f}fps "
                f"FOURCC={fourcc_str(cap)}  {'OK' if ok else '불일치'}"
            )
            bad += 0 if ok else 1
            cap.release()
        if has_v4l2:
            s = ctrl_summary(node)
            if s:
                say(f"   노출/WB: {s}")
            if a.formats:
                say("   --- v4l2-ctl --list-formats-ext")
                say("   " + v4l2("-d", node, "--list-formats-ext")[1].replace("\n", "\n   "))
            if a.ctrls:
                say("   --- v4l2-ctl --list-ctrls")
                say("   " + v4l2("-d", node, "--list-ctrls")[1].replace("\n", "\n   "))
    say("")
    say("robot.env 에는 by-id(또는 by-path) 의 video-index0 경로를 쓴다. 노트북 내장 웹캠은 빼고 고를 것.")
    return 0 if bad == 0 else 1


# ----------------------------------------------------------------------------- check


def focus_state_path() -> Path:
    return day_dir() / "focus_peaks.json"  # 오늘 것만: 다른 날·다른 장면의 최고값은 기준이 못 됨


def focus_key(cam: str, w, h) -> str:
    return f"{os.path.realpath(cam)}|{w}x{h}"


def cmd_check(a) -> int:
    r = open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    if r is None:
        return 2
    cap, _ = r
    try:
        aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rep_fps, fcc = cap.get(cv2.CAP_PROP_FPS), fourcc_str(cap)
        say(f"{a.cam}: 요청 {a.width}x{a.height}@{a.fps} → 보고 {aw}x{ah}@{rep_fps:.1f} FOURCC={fcc}")
        say(f"자동 노출 안정화 {a.warmup:g}s, 측정 {a.seconds:g}s ...")
        warmup(cap, a.warmup)
        period = 1.0 / a.fps
        stamps, samples, dup, fails, last, prev_small = [], [], 0, 0, None, None
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            ok, f = cap.read()
            now = time.time()
            if not ok or f is None:
                fails += 1
                if fails > 30:
                    break
                continue
            stamps.append(now)
            small = f[::8, ::8]
            if prev_small is not None and np.array_equal(small, prev_small):
                dup += 1
            prev_small = small
            if len(stamps) % 10 == 1:
                samples.append(metrics(f, a.roi))
            last = f
    finally:
        cap.release()

    if last is None or len(stamps) < 2:
        say("[FAIL] 측정 중 프레임이 거의 안 나옴.")
        say(BUSY_HINT)
        return 1
    dts = np.diff(stamps)
    meas_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    drops = int(sum(max(0, round(dt / period) - 1) for dt in dts if dt > 1.5 * period))
    m = {k: float(np.median([s[k] for s in samples])) for k in samples[0]}
    shape_ok = last.shape[1::-1] == (a.width, a.height)

    results = []

    def verdict(level: str, item: str, msg: str):
        results.append(level)
        say(f"  [{level:4}] {item:10} {msg}")

    say("")
    say("결과")
    verdict(
        "PASS" if shape_ok else "FAIL",
        "해상도",
        f"실제 프레임 {last.shape[1]}x{last.shape[0]} (목표 {a.width}x{a.height}, 정확히 같아야 함 — lerobot 은 다르면 에러)",
    )
    verdict(
        "PASS" if meas_fps >= FPS_MIN else "WARN",
        "fps",
        f"실측 {meas_fps:.1f} (목표 ≥ {FPS_MIN:.0f}), 카메라 보고값 {rep_fps:.1f}",
    )
    verdict(
        "PASS" if drops == 0 and fails == 0 else "WARN",
        "드랍",
        f"추정 누락 {drops}프레임, 읽기 실패 {fails}, 중복 프레임 {dup} / {len(stamps)}프레임 "
        f"(간격 최대 {dts.max() * 1000:.0f}ms)",
    )
    verdict(
        "PASS" if m["sat_pct"] < SAT_MAX_PCT else "WARN",
        "포화",
        f"{m['sat_pct']:.2f}% (목표 < {SAT_MAX_PCT}%) — 하얗게 날아간 픽셀 비율",
    )
    say(
        f"  [INFO] {'밝기':10} 평균 {m['brightness']:.0f}/255, 암부(≤5) {m['dark_pct']:.1f}%, 채도 평균 {m['colorfulness']:.0f}"
    )

    peaks = {}
    with contextlib.suppress(OSError, ValueError):
        peaks = json.loads(focus_state_path().read_text())
    ref = a.sharp_ref or peaks.get(focus_key(a.cam, a.width, a.height))
    sc = m["sharp_center"]
    if ref:
        lvl = "PASS" if sc >= SHARP_REL * ref else "WARN"
        why = f"focus 최고값 {ref:.0f} 의 {sc / ref * 100:.0f}% (기준 ≥ {SHARP_REL * 100:.0f}%, 같은 대상일 때만 의미 있음)"
    else:
        lvl = "PASS" if sc >= SHARP_FLOOR else "WARN"
        why = f"절대 하한 {SHARP_FLOOR:.0f} 기준 (focus 를 먼저 돌리면 그 최고값 대비로 판정)"
    verdict(lvl, "선명도", f"중앙 {a.roi:.0%} 영역 {sc:.0f}, 전체 {m['sharp']:.0f} — {why}")

    s = ctrl_summary(a.cam)
    if s:
        say(f"  [INFO] {'노출/WB':10} {s}")
        if meas_fps < FPS_MIN:
            say(
                "         fps 가 낮으면: 노출 시간이 1/fps 보다 긴지(exposure_time_absolute 단위 100µs → 30fps 는 ≤ 333),"
            )
            say(
                "         exposure_dynamic_framerate=1(어두우면 fps 를 낮춤)인지, 한 USB 허브에 여러 대면 --fourcc MJPG 를 시도."
            )

    out = day_dir() / f"check_{slug_of(a.cam)}_{a.width}x{a.height}_{datetime.now():%H%M%S}.png"
    cv2.imwrite(str(out), last)
    ann = last.copy()
    x0, y0, cw, ch = center_box(ann.shape, a.roi)
    cv2.rectangle(ann, (x0, y0), (x0 + cw, y0 + ch), (0, 255, 255), 1)
    ann_out = out.with_name(out.stem + "_roi.png")
    cv2.imwrite(str(ann_out), ann)
    say("")
    say(f"저장: {out}")
    say(f"      {ann_out} (노란 상자 = 선명도 측정 영역)")
    final = "FAIL" if "FAIL" in results else ("WARN" if "WARN" in results else "PASS")
    say(f"종합: {final}")
    return 0 if final == "PASS" else 1


# ----------------------------------------------------------------------------- focus


def cmd_focus(a) -> int:
    r = open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    if r is None:
        return 2
    cap, _ = r
    use_win = has_display() and not a.no_window
    win = f"camera_check focus [{slug_of(a.cam)}]"
    peak, smooth, last_print, frame = 0.0, None, 0.0, None
    t0 = time.time()
    say(
        f"초점 맞추기: {a.cam} {a.width}x{a.height}. 컵 거리(손목캠에서 5~15cm)에 무늬 있는 대상(인쇄 글씨)을 두고"
    )
    say("렌즈를 돌려 숫자가 더 안 오르는 지점(최고값)에서 멈춘다. 빈 면을 보면 초점이 맞아도 숫자가 낮다.")
    say("창: r=최고값 초기화, s=스냅샷, q/ESC=종료" if use_win else "콘솔 모드 (창 없음): Ctrl+C 로 종료")
    try:
        while True:
            if a.seconds and time.time() - t0 > a.seconds:
                break
            ok, frame_new = cap.read()
            if not ok:
                continue
            frame = frame_new
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x0, y0, cw, ch = center_box(frame.shape, a.roi)
            v = sharpness(gray[y0 : y0 + ch, x0 : x0 + cw])
            smooth = v if smooth is None else 0.75 * smooth + 0.25 * v  # 프레임마다 튀므로 평활
            peak = max(peak, smooth)
            if use_win:
                view = frame.copy()
                cv2.rectangle(view, (x0, y0), (x0 + cw, y0 + ch), (0, 255, 255), 1)
                edges = cv2.cvtColor(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S)), cv2.COLOR_GRAY2BGR)
                canvas = tile([("live", view), ("edges", edges)], height=max(360, frame.shape[0]))
                bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
                frac = min(1.0, smooth / max(peak, 1e-6))
                cv2.rectangle(
                    bar,
                    (0, 0),
                    (int(frac * canvas.shape[1]), 33),
                    (0, 140, 0) if frac > 0.95 else (0, 90, 160),
                    -1,
                )
                cv2.putText(
                    bar,
                    f"sharpness {smooth:7.1f}  peak {peak:7.1f}  ({frac * 100:3.0f}% of peak)  [r]eset [s]ave [q]uit",
                    (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                try:
                    cv2.imshow(win, np.vstack([bar, canvas]))
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    say("창을 띄울 수 없음 (headless OpenCV?) → 콘솔 모드로 전환")
                    use_win, key = False, 255
                if key == ord("r"):
                    peak, smooth = 0.0, None
                elif key == ord("s"):
                    p = day_dir() / f"focus_{slug_of(a.cam)}_{datetime.now():%H%M%S}.png"
                    cv2.imwrite(str(p), frame)
                    say(f"저장 {p} (선명도 {smooth:.0f})")
                elif key in (ord("q"), 27):
                    break
            if not use_win and time.time() - last_print >= 0.5:
                last_print = time.time()
                frac = min(1.0, smooth / max(peak, 1e-6))
                say(
                    f"선명도 {smooth:7.1f}  최고 {peak:7.1f}  |{'#' * int(frac * 30):<30}| {frac * 100:3.0f}%"
                )
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if use_win:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except cv2.error:
                pass
    say("")
    say(f"최고 선명도: {peak:.1f} (중앙 {a.roi:.0%} 영역)")
    if peak > 0:
        try:
            st = json.loads(focus_state_path().read_text()) if focus_state_path().exists() else {}
        except ValueError:
            st = {}
        st[focus_key(a.cam, a.width, a.height)] = round(peak, 1)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        focus_state_path().write_text(json.dumps(st, indent=1))
        say(f"→ check 가 이 값의 {SHARP_REL:.0%} 이상을 PASS 로 본다 ({focus_state_path()})")
    if frame is not None:
        p = day_dir() / f"focus_{slug_of(a.cam)}_last.png"
        cv2.imwrite(str(p), frame)
        say(f"마지막 프레임: {p}")
    return 0


# ----------------------------------------------------------------------------- exposure


def cmd_exposure(a) -> int:
    if not shutil.which("v4l2-ctl"):
        say("[오류] v4l2-ctl 없음: sudo apt install v4l-utils")
        return 2
    ctrls = read_ctrls(a.cam)
    if not ctrls:
        say(f"[오류] {a.cam} 의 컨트롤을 못 읽음 (경로 확인, video-index0 인지)")
        say(BUSY_HINT)
        return 2
    n_ae, n_exp, n_wba, n_wbt, n_dyn = (
        pick(ctrls, k) for k in ("auto_exposure", "exposure", "wb_auto", "wb_temp", "dyn_fps")
    )
    say(f"현재: {ctrl_summary(a.cam)}")
    if a.sweep:
        return exposure_sweep(a, ctrls, n_ae, n_exp, n_wba, n_wbt, n_dyn)

    sets: list[tuple[str, int]] = []
    if a.reset:
        for n in (n_ae, n_wba):
            if n:
                sets.append((n, ctrls[n].get("default", 1)))
        if n_dyn:
            sets.append((n_dyn, ctrls[n_dyn].get("default", 0)))
    else:
        if a.exposure is not None:
            if not (n_ae and n_exp):
                say("[경고] 이 카메라는 수동 노출 컨트롤이 없음")
            else:
                manual = next((k for k, v in ctrls[n_ae]["menu"].items() if "manual" in v.lower()), 1)
                lo, hi = ctrls[n_exp].get("min"), ctrls[n_exp].get("max")
                if isinstance(lo, int) and isinstance(hi, int) and not lo <= a.exposure <= hi:
                    say(f"[경고] exposure {a.exposure} 가 범위 [{lo}, {hi}] 밖 → 카메라가 잘라낼 수 있음")
                if a.exposure * 100e-6 > 1.0 / a.fps:
                    say(
                        f"[경고] exposure {a.exposure} = {a.exposure / 10:.1f}ms > 1/{a.fps:.0f}s → fps 가 떨어진다 "
                        f"(단위 100µs, {a.fps:.0f}fps 는 ≤ {int(1e4 / a.fps)})"
                    )
                sets += [(n_ae, manual), (n_exp, a.exposure)]
                if n_dyn:
                    sets.append((n_dyn, 0))  # 어두워도 fps 를 낮추지 않게
        if a.wb_temp is not None:
            if not (n_wba and n_wbt):
                say("[경고] 이 카메라는 수동 화이트밸런스 컨트롤이 없음")
            else:
                sets += [(n_wba, 0), (n_wbt, a.wb_temp)]
        for kv in a.set or []:
            k, _, v = kv.partition("=")
            sets.append((k.strip(), int(v)))
    if not sets:
        say("바꿀 값이 없음 (--exposure / --wb-temp / --set name=val / --reset). 현재 값만 출력하고 끝.")
        return 0

    restore = [(n, ctrls[n]["value"]) for n, _ in sets if n in ctrls and "value" in ctrls[n]]
    r = open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    if r is None:
        return 2
    cap, _ = r
    try:
        before = warmup(cap, 1.0)
        cmds = []
        for n, v in sets:  # 하나씩: auto 를 끈 뒤에 값을 넣어야 먹는 카메라가 있다
            cmd = f"v4l2-ctl -d {a.cam} --set-ctrl={n}={v}"
            rc, out = v4l2("-d", a.cam, f"--set-ctrl={n}={v}")
            say(f"$ {cmd}" + ("" if rc == 0 else f"   → 실패: {out}"))
            cmds.append(cmd)
        after = warmup(cap, a.settle)
    finally:
        cap.release()
    say(f"적용 후: {ctrl_summary(a.cam)}")

    d = day_dir()
    stamp = f"{slug_of(a.cam)}_{datetime.now():%H%M%S}"
    if before is not None and after is not None:
        mb, ma = metrics(before), metrics(after)
        say(f"전: {fmt_metrics(mb)}")
        say(f"후: {fmt_metrics(ma)}")
        cv2.imwrite(str(d / f"exposure_{stamp}_before.png"), before)
        cv2.imwrite(str(d / f"exposure_{stamp}_after.png"), after)
        sub = [
            f"bright {mb['brightness']:.0f} sat {mb['sat_pct']:.1f}% color {mb['colorfulness']:.0f}",
            f"bright {ma['brightness']:.0f} sat {ma['sat_pct']:.1f}% color {ma['colorfulness']:.0f}",
        ]
        label = "after (reset to auto)" if a.reset else f"after exp={a.exposure} wb={a.wb_temp}"
        cv2.imwrite(
            str(d / f"exposure_{stamp}_compare.png"),
            tile([("before", before), (label, after)], height=max(240, before.shape[0]), sub=sub),
        )
        say(f"저장: {d}/exposure_{stamp}_{{before,after,compare}}.png")
    if a.reset:
        say("자동으로 되돌림 (cam_setup .sh 는 그대로 둠).")
        return 0
    # 되돌릴 땐 역순: 값 먼저, auto 는 마지막 (auto 가 켜진 상태에선 값 설정이 거부됨)
    undo = " ; ".join(f"v4l2-ctl -d {a.cam} --set-ctrl={n}={v}" for n, v in reversed(restore))
    sh = d / f"cam_setup_{slug_of(a.cam)}.sh"
    sh.write_text(
        "#!/bin/bash\n# camera_check exposure 가 만든 파일. USB 재연결·재부팅 후 lerobot 실행 직전에 다시 실행.\n"
        + "\n".join(cmds)
        + f"\n# 원래대로: {undo}\n"
    )
    say("")
    say(
        f"위 명령을 {sh} 에 저장. ※ V4L2 설정은 USB 재연결·재부팅 때 초기화된다 → 매 세션 lerobot 실행 직전에 bash 로 다시 실행."
    )
    say("※ OpenCV 로 카메라를 다시 열어도 값은 유지된다. 학습 데이터 수집 때와 같은 값을 쓰는 게 원칙.")
    return 0


def exposure_sweep(a, ctrls, n_ae, n_exp, n_wba, n_wbt, n_dyn) -> int:
    """노출값 후보를 차례로 적용·촬영해 한 장으로 비교. 끝나면 원래 값으로 복구 (고르는 용도)."""
    if not (n_ae and n_exp):
        say("[오류] 이 카메라는 수동 노출 컨트롤이 없음")
        return 1
    try:
        values = [int(v) for v in a.sweep.split(",") if v.strip()]
    except ValueError:
        say("[오류] --sweep 은 정수 목록 (예 50,100,200,300)")
        return 1
    manual = next((k for k, v in ctrls[n_ae]["menu"].items() if "manual" in v.lower()), 1)
    touched = [n for n in (n_ae, n_exp, n_dyn, n_wba, n_wbt) if n]
    restore = [(n, ctrls[n]["value"]) for n in touched if "value" in ctrls[n]]
    pre = [(n_ae, manual)] + ([(n_dyn, 0)] if n_dyn else [])
    if a.wb_temp is not None and n_wba and n_wbt:
        pre += [(n_wba, 0), (n_wbt, a.wb_temp)]
    r = open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    if r is None:
        return 2
    cap, _ = r
    shots = []
    try:
        for n, v in pre:
            v4l2("-d", a.cam, f"--set-ctrl={n}={v}")
        for v in values:
            v4l2("-d", a.cam, f"--set-ctrl={n_exp}={v}")
            f = warmup(cap, a.settle)
            if f is None:
                continue
            m = metrics(f)
            slow = "  (fps 떨어짐!)" if v * 100e-6 > 1.0 / a.fps else ""
            say(f"exposure {v:5d}: {fmt_metrics(m)}{slow}")
            shots.append((v, f, m))
    finally:
        cap.release()
        for n, v in reversed(restore):
            v4l2("-d", a.cam, f"--set-ctrl={n}={v}")
        say(f"원래 설정으로 복구: {ctrl_summary(a.cam)}")
    if not shots:
        return 1
    out = day_dir() / f"exposure_sweep_{slug_of(a.cam)}_{datetime.now():%H%M%S}.png"
    cv2.imwrite(
        str(out),
        tile(
            [(f"exp={v}" + (" SLOW" if v * 100e-6 > 1.0 / a.fps else ""), f) for v, f, _ in shots],
            height=max(240, shots[0][1].shape[0]),
            sub=[
                f"bright {m['brightness']:.0f} sat {m['sat_pct']:.1f}% color {m['colorfulness']:.0f}"
                for _, _, m in shots
            ],
        ),
    )
    say(f"비교: {out}")
    say("포화가 거의 없으면서(< 2%) 컵·물 색이 가장 잘 보이는 값을 골라 --exposure N 으로 다시 실행.")
    return 0


# ----------------------------------------------------------------------------- capture / compare


def safe_label(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s or "capture"


def cmd_capture(a) -> int:
    label = safe_label(a.label)
    r = open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    if r is None:
        return 2
    cap, _ = r
    d = day_dir()
    existing = [
        int(m.group(1))
        for p in d.glob(f"{label}_*.png")
        if (m := re.fullmatch(rf"{re.escape(label)}_(\d+)\.png", p.name))
    ]
    n0 = max(existing, default=0) + 1
    saved = []
    try:
        warmup(cap, a.warmup)
        for i in range(a.count):
            frame = None
            t_next = time.time() + (a.interval if i else 0)
            while frame is None or time.time() < t_next:
                ok, f = cap.read()
                if ok:
                    frame = f
            p = d / f"{label}_{n0 + i:02d}.png"
            cv2.imwrite(str(p), frame)
            saved.append(p)
            say(f"저장 {p}  ({frame.shape[1]}x{frame.shape[0]}; {fmt_metrics(metrics(frame))})")
    finally:
        cap.release()
    with open(d / "captures.jsonl", "a") as fh:
        for p in saved:
            fh.write(
                json.dumps(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "label": label,
                        "file": p.name,
                        "cam": a.cam,
                        "size": [a.width, a.height],
                        "ctrls": ctrl_summary(a.cam),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return 0


def latest_capture(label: str, day: str | None) -> Path | None:
    days = (
        [OUT_ROOT / day]
        if day
        else sorted((p for p in OUT_ROOT.glob("????-??-??") if p.is_dir()), reverse=True)
    )
    for d in days:
        files = sorted(
            d.glob(f"{label}_[0-9]*.png"),
            key=lambda p: int(re.sub(r"\D", "", p.stem.rsplit("_", 1)[-1]) or 0),
        )
        if files:
            return files[-1]
    return None


def parse_box(s: str | None, shape) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    if not s:
        return center_box(shape, 0.5)
    x, y, bw, bh = (min(1.0, max(0.0, float(v))) for v in s.split(","))
    x0, y0 = min(int(x * w), w - 1), min(int(y * h), h - 1)
    return x0, y0, max(1, min(int(bw * w), w - x0)), max(1, min(int(bh * h), h - y0))


def cmd_compare(a) -> int:
    imgs, names = [], []
    for lab in a.labels:
        p = (
            Path(lab)
            if lab.endswith(".png") and Path(lab).exists()
            else latest_capture(safe_label(lab), a.day)
        )
        img = cv2.imread(str(p)) if p else None
        if img is None:
            say(
                f"[오류] '{lab}' 캡처를 못 찾음 ({OUT_ROOT}/<날짜>/{safe_label(lab)}_NN.png). capture 를 먼저 실행."
            )
            return 1
        imgs.append(img)
        names.append((lab if not lab.endswith(".png") else Path(lab).stem, p))
        say(f"{lab}: {p}")
    h = max(i.shape[0] for i in imgs)
    full_h = max(360, h)
    sub_full = [f"{p.name}  {i.shape[1]}x{i.shape[0]}" for (_, p), i in zip(names, imgs, strict=True)]
    boxed = []
    for img in imgs:
        b = img.copy()
        x, y, bw, bh = parse_box(a.crop, img.shape)
        cv2.rectangle(b, (x, y), (x + bw - 1, y + bh - 1), (0, 255, 255), 1)
        boxed.append(b)
    row1 = tile([(n, b) for (n, _), b in zip(names, boxed, strict=True)], height=full_h, sub=sub_full)
    crops = []
    for img in imgs:
        x, y, bw, bh = parse_box(a.crop, img.shape)
        crops.append(img[y : y + bh, x : x + bw])
    zoom = round(full_h / crops[0].shape[0], 1)
    row2 = tile([(f"{n} (crop x{zoom})", c) for (n, _), c in zip(names, crops, strict=True)], height=full_h)
    title = np.full((30, max(row1.shape[1], row2.shape[1]), 3), 40, np.uint8)
    cv2.putText(
        title,
        f"camera_check compare  {datetime.now():%Y-%m-%d %H:%M}  (top: full frame, yellow = crop | bottom: zoomed crop, nearest-neighbour)",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    canvas = stack_rows([title, row1, row2])
    out = (
        day_dir(a.day)
        / f"compare_{'_vs_'.join(safe_label(n) for n, _ in names)[:120]}_{datetime.now():%H%M%S}.png"
    )
    cv2.imwrite(str(out), canvas)
    say("")
    say(f"비교 PNG: {out}  ({canvas.shape[1]}x{canvas.shape[0]})")
    say("이 파일을 Claude 에게 보내면 된다. 컵이 중앙에 없으면 --crop x,y,w,h (0~1 비율) 로 자를 영역 지정.")
    return 0


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def cam_args(p, w=320, h=240):
        p.add_argument(
            "--cam", required=True, help="/dev/v4l/by-id/...-video-index0 (또는 /dev/videoN, 숫자)"
        )
        p.add_argument("--width", type=int, default=w, help=f"기본 {w} (top 은 640)")
        p.add_argument("--height", type=int, default=h, help=f"기본 {h} (top 은 480)")
        p.add_argument("--fps", type=float, default=30)
        p.add_argument(
            "--fourcc",
            default=None,
            help="MJPG / YUYV (기본: 카메라 자동). robot.env 의 CAM_FOURCC 와 맞출 것",
        )

    p = sp.add_parser("list", help="비디오 장치 목록과 학습 해상도로 열리는지")
    p.add_argument("--fourcc", default=None)
    p.add_argument("--formats", action="store_true", help="v4l2-ctl --list-formats-ext 도 출력")
    p.add_argument("--ctrls", action="store_true", help="v4l2-ctl --list-ctrls 도 출력")
    p.set_defaults(fn=cmd_list)

    p = sp.add_parser("check", help="fps·드랍·밝기·포화·선명도 측정 + PNG")
    cam_args(p)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--warmup", type=float, default=1.5, help="측정 전 버리는 시간 (자동 노출 안정화)")
    p.add_argument("--roi", type=float, default=0.4, help="선명도 중앙 영역 비율")
    p.add_argument("--sharp-ref", type=float, default=None, help="비교 기준 선명도 (기본: 오늘 focus 최고값)")
    p.set_defaults(fn=cmd_check)

    p = sp.add_parser("focus", help="렌즈 초점용 실시간 선명도")
    cam_args(p)
    p.add_argument("--roi", type=float, default=0.4)
    p.add_argument("--no-window", action="store_true", help="창 없이 콘솔에 0.5초마다 출력")
    p.add_argument("--seconds", type=float, default=0, help="이 시간 뒤 자동 종료 (0 = q/Ctrl+C 까지)")
    p.set_defaults(fn=cmd_focus)

    p = sp.add_parser("exposure", help="v4l2-ctl 로 수동 노출/WB 적용 + 전후 PNG")
    cam_args(p)
    p.add_argument(
        "--exposure", type=int, default=None, help="exposure_time_absolute (단위 100µs; 30fps 는 ≤ 333)"
    )
    p.add_argument("--wb-temp", type=int, default=None, help="white_balance_temperature (K, 예 4600)")
    p.add_argument(
        "--set", action="append", metavar="NAME=VAL", help="기타 컨트롤 (예 --set gain=0), 여러 번 가능"
    )
    p.add_argument("--reset", action="store_true", help="자동 노출/WB 로 되돌림 (각 컨트롤의 default)")
    p.add_argument("--settle", type=float, default=1.5, help="적용 후 기다리는 시간")
    p.add_argument(
        "--sweep",
        default=None,
        metavar="V1,V2,...",
        help="노출값 여러 개를 차례로 찍어 한 장에 비교 (예 50,100,200,300). 끝나면 원래 설정으로 되돌림",
    )
    p.set_defaults(fn=cmd_exposure)

    p = sp.add_parser("capture", help="라벨 붙여 PNG 저장 (수위 시험)")
    cam_args(p)
    p.add_argument("--label", required=True, help="예 empty, level_minus1cm, level_target, level_plus1cm")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--interval", type=float, default=0.3, help="장 사이 간격(s)")
    p.add_argument("--warmup", type=float, default=1.5)
    p.set_defaults(fn=cmd_capture)

    p = sp.add_parser("compare", help="라벨별 최신 캡처를 나란히 + 확대 crop 줄 → 비교 PNG 1장")
    p.add_argument("--labels", nargs="+", required=True, help="라벨 (또는 PNG 경로) 여러 개")
    p.add_argument(
        "--crop", default=None, help="확대할 영역 x,y,w,h (0~1 비율, 기본 중앙 50%%: 0.25,0.25,0.5,0.5)"
    )
    p.add_argument("--day", default=None, help="YYYY-MM-DD (기본: 라벨마다 가장 최근 날짜)")
    p.set_defaults(fn=cmd_compare)

    a = ap.parse_args()
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        say("\n중단됨")
        return 130
    except Exception as e:  # 현장에서 트레이스백 대신 원인 한 줄
        say(f"[오류] {type(e).__name__}: {e}")
        if os.environ.get("CAMERA_CHECK_DEBUG"):
            raise
        say("(자세한 트레이스백: CAMERA_CHECK_DEBUG=1 로 다시 실행)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
