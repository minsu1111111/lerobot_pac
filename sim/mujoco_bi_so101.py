"""양팔 SO-101 MuJoCo 가짜 로봇 (관절 수준). 정책에 이미지를 주지 않는다 — 관절 궤적 확인·롤아웃 루프 시험용.

  sim = MujocoBiSO101(render="offscreen", video_path="out.mp4", physics=True)
  obs = sim.reset(state12)          # LeRobot 단위 12차원 (데이터셋 observation.state 와 같은 순서)
  obs = sim.step(action12)          # 위치 명령 → 1/fps 초 진행 → 시뮬 관절값 12차원 (LeRobot 단위)
  sim.close()

단위 변환 (LeRobot so_follower, use_degrees=True ↔ MJCF so101_new_calib, 라디안):
  몸통 5관절: LeRobot DEGREES = (raw - 보정범위 중앙) * 360/4095 → 0° = 범위 중앙.
             new_calib MJCF 도 0 rad = 범위 중앙이라 rad = deg2rad(deg), 부호 반전 없음 (영상·FK 로 확인).
  gripper   : RANGE_0_100 (0 = 닫힘, 100 = 열림) → MJCF [-0.1745, 1.7453] rad 에 선형 (0 → -0.1745 = 닫힘).
  MJCF 관절 범위를 넘는 값은 잘린다 (예: 오른팔 wrist_roll 데이터 최대 167.7° > MJCF 상한 162.8°).

렌더링 (이 머신, Intel iGPU, 2026-10-03 확인):
  MUJOCO_GL=egl    실패 (EGLError, 사용자 계정이 render 그룹 아님)
  MUJOCO_GL=glfw   동작, 640x480 약 70 fps (DISPLAY 필요; 오프스크린도 숨은 창으로 동작)
  MUJOCO_GL=osmesa 동작하지만 640x480 약 2.5 fps (디스플레이 없을 때만)
  → MUJOCO_GL 이 비어 있으면 DISPLAY 가 있을 때 glfw, 없으면 osmesa 를 쓴다. mujoco import 전에 정해야 한다.
"""

import os
import sys
import time
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if os.environ.get("DISPLAY") else "osmesa"

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bi_so101_scene import BOTTLE, CUP, JOINTS, SIDES, build_model  # noqa: E402

NAMES = [f"{s}_{j}" for s in SIDES for j in JOINTS]  # LeRobot 12차원 순서 (.pos 생략)
SIGN = np.ones(12)  # 몸통 관절 부호 (전부 +1 로 확인됨). 그리퍼 칸은 쓰지 않음
GRIP = [5, 11]
BODY = [i for i in range(12) if i not in GRIP]
GRIP_LO, GRIP_HI = -0.17453297762778586, 1.7453291995659765  # MJCF gripper 범위 (rad)

# 소품 잡기 판정 (LeRobot 그리퍼 값): 열림(>OPEN) 후 닫힘(<CLOSE) 순간 물체가 가까우면 붙임, 다시 열리면 놓음
GRIP_OPEN, GRIP_CLOSE, GRASP_DIST = 80.0, 60.0, 0.08
# 물리 모드 기본 kp = joints_properties.xml 의 17.8 (kv=MJCF 2.731). ep0/ep37 에서 시뮬 상태가 실제 기록 상태에
# 가장 가까웠다 (MJCF 기본 998.22 는 명령을 거의 지연 없이 따라가 실제보다 너무 빠름).
KP_DEFAULT = 17.8
PROPS = {"bottle": ("right", BOTTLE), "cup": ("left", CUP)}


def lerobot_to_qpos(x12, model: mujoco.MjModel | None = None, prefix: str = "") -> np.ndarray:
    """LeRobot 단위 12차원 → MJCF 라디안 12차원. model 을 주면 관절 범위로 자른다."""
    x = np.asarray(x12, dtype=np.float64)
    q = np.empty(12)
    q[BODY] = np.deg2rad(SIGN[BODY] * x[BODY])
    q[GRIP] = GRIP_LO + x[GRIP] / 100.0 * (GRIP_HI - GRIP_LO)
    if model is not None:
        rng = np.array([model.joint(prefix + n).range for n in NAMES])
        q = np.clip(q, rng[:, 0], rng[:, 1])
    return q


def qpos_to_lerobot(q12) -> np.ndarray:
    """MJCF 라디안 12차원 → LeRobot 단위 12차원 (float32)."""
    q = np.asarray(q12, dtype=np.float64)
    x = np.empty(12)
    x[BODY] = SIGN[BODY] * np.rad2deg(q[BODY])
    x[GRIP] = (q[GRIP] - GRIP_LO) / (GRIP_HI - GRIP_LO) * 100.0
    return x.astype(np.float32)


_FONT = None


def _font(size=18):
    """한글 폰트 (Noto CJK). 없으면 None → cv2 영문 글꼴로 대체."""
    global _FONT
    if _FONT is None:
        _FONT = False
        try:
            from PIL import ImageFont

            for f in (
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            ):
                if os.path.exists(f):
                    _FONT = ImageFont.truetype(f, size)
                    break
        except ImportError:
            pass
    return _FONT or None


def _draw_text(img, text):
    font = _font()
    if font is not None:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for i, line in enumerate(text.split("\n")):
            dr.text(
                (10, 8 + 24 * i), line, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)
            )
        return np.asarray(im)
    import cv2

    img = img.copy()
    for i, line in enumerate(text.split("\n")):
        y = 24 + 22 * i
        line = line.encode("ascii", "replace").decode()
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _paste_inset(img, inset, frac=0.3, margin=8):
    """inset 을 img 폭의 frac 크기로 줄여 오른쪽 위에 흰 테두리와 함께 붙인다."""
    import cv2

    h, w = img.shape[:2]
    iw = int(w * frac)
    ih = int(inset.shape[0] * iw / inset.shape[1])
    small = cv2.resize(np.ascontiguousarray(inset), (iw, ih), interpolation=cv2.INTER_AREA)
    out = img.copy()
    x0, y0 = w - iw - margin, margin
    out[y0 - 2 : y0 + ih + 2, x0 - 2 : x0 + iw + 2] = 255
    out[y0 : y0 + ih, x0 : x0 + iw] = small
    return out


def _pose(d, site_id):
    return d.site_xpos[site_id].copy(), d.site_xmat[site_id].reshape(3, 3).copy()


class MujocoBiSO101:
    def __init__(
        self,
        render: str = "none",
        video_path=None,
        fps: int = 30,
        physics: bool = True,
        spacing: float = 0.45,
        ghost: bool = False,
        props: bool = True,
        camera: str = "front",
        width: int = 640,
        height: int = 480,
        realtime: bool = True,
        kp: float | None = KP_DEFAULT,
        **scene_kw,
    ):
        """render: none | window(mujoco.viewer) | offscreen(Renderer; video_path 주면 mp4).
        physics=False: qpos = 명령 (순수 시각화). kp: 위치 액추에이터 게인 (None = MJCF 998.22).
        scene_kw: bi_so101_scene.build_spec 인자 (yaw_deg, x_offset, contacts, kv, so101=자산 폴더 ...)."""
        assert render in ("none", "window", "offscreen")
        self.fps, self.physics, self.ghost, self.realtime = fps, physics, ghost, realtime
        self.m = build_model(spacing=spacing, ghost=ghost, props=props, kp=kp, **scene_kw)
        self.d = mujoco.MjData(self.m)
        self.n_sub = max(1, round(1.0 / (fps * self.m.opt.timestep)))
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in NAMES])
        self.vadr = np.array([self.m.joint(n).dofadr[0] for n in NAMES])
        self.act = np.array([self.m.actuator(n).id for n in NAMES])
        if ghost:
            self.gq = np.array([self.m.joint("ghost_" + n).qposadr[0] for n in NAMES])
            self.gv = np.array([self.m.joint("ghost_" + n).dofadr[0] for n in NAMES])
            self.ga = np.array([self.m.actuator("ghost_" + n).id for n in NAMES])
            self._ghost_q = None
        self.text = ""  # 오프스크린 프레임 왼쪽 위에 찍을 글자 (여러 줄은 \n, 한글 가능)
        self.inset = None  # 오른쪽 위에 작게 붙일 카메라 영상 (uint8 HWC), 롤아웃 영상용
        self.frames = 0

        # 소품
        self.props = {}
        if props:
            for name, (side, p) in PROPS.items():
                self.props[name] = dict(
                    mocap=self.m.body(name).mocapid[0],
                    site=self.m.site(f"{side}_gripperframe").id,
                    grip=GRIP[SIDES.index(side)],
                    half=p["half"],
                    held=None,
                    armed=False,
                )

        # 렌더러
        self.viewer = self.renderer = self.writer = None
        self.camera = camera
        if render == "window":
            self.viewer = mujoco.viewer.launch_passive(
                self.m, self.d, show_left_ui=False, show_right_ui=False
            )
        if render == "offscreen" or video_path is not None:
            self.renderer = mujoco.Renderer(self.m, height, width)
        if video_path is not None:
            import imageio.v2 as imageio

            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                str(video_path),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=16,
                ffmpeg_log_level="error",
            )
        self._t_last = None

    # ---------- 상태 입출력 ----------
    def get_state(self) -> np.ndarray:
        return qpos_to_lerobot(self.d.qpos[self.qadr])

    def reset(self, state12) -> np.ndarray:
        q = lerobot_to_qpos(state12, self.m)
        mpos, mquat = self.d.mocap_pos.copy(), self.d.mocap_quat.copy()  # 소품 배치는 유지
        mujoco.mj_resetData(self.m, self.d)
        self.d.mocap_pos[:], self.d.mocap_quat[:] = mpos, mquat
        self.d.qpos[self.qadr] = q
        self.d.ctrl[self.act] = q
        if self.ghost:
            if self._ghost_q is None:  # set_ghost 를 먼저 안 불렀으면 같은 자세로
                self._ghost_q = q
            self._apply_ghost()
        mujoco.mj_forward(self.m, self.d)
        for p in self.props.values():
            p["held"], p["armed"] = None, False
        self._update_props()
        self.frames = 0
        self._t_last = None
        self._render()
        return self.get_state()

    def step(self, action12) -> np.ndarray:
        q = lerobot_to_qpos(action12, self.m)
        if self.physics:
            self.d.ctrl[self.act] = q
            for _ in range(self.n_sub):
                mujoco.mj_step(self.m, self.d)
        else:  # 순수 시각화: 명령 = 자세
            self.d.qpos[self.qadr] = q
            self.d.qvel[self.vadr] = 0
            self.d.time += 1.0 / self.fps
        if self.ghost:
            self._apply_ghost()
        mujoco.mj_forward(self.m, self.d)
        self._update_props()
        self.frames += 1
        self._render()
        return self.get_state()

    def set_ghost(self, x12):
        """반투명 리그 자세 (LeRobot 단위). ghost=True 일 때만. 다음 step/reset 렌더에 반영."""
        self._ghost_q = lerobot_to_qpos(x12, self.m, "ghost_")

    def _apply_ghost(self):
        if self._ghost_q is not None:
            self.d.qpos[self.gq] = self._ghost_q
            self.d.qvel[self.gv] = 0
            self.d.ctrl[self.ga] = self._ghost_q

    # ---------- 소품 (컵·물통) ----------
    def place_props_from_trajectory(self, traj):
        """궤적(T,12, LeRobot 단위)을 순기구학으로 훑어 처음 잡는 위치에 소품을 세워 둔다."""
        if not self.props:
            return
        d = mujoco.MjData(self.m)
        traj = np.asarray(traj)
        for p in self.props.values():
            armed, pos = False, None
            for x in traj:
                g = x[p["grip"]]
                armed = armed or g > GRIP_OPEN
                if armed and g < GRIP_CLOSE:
                    d.qpos[self.qadr] = lerobot_to_qpos(x, self.m)
                    mujoco.mj_kinematics(self.m, d)
                    pos = d.site_xpos[p["site"]].copy()
                    break
            if pos is not None:
                self.d.mocap_pos[p["mocap"]] = [pos[0], pos[1], p["half"]]
                self.d.mocap_quat[p["mocap"]] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def _update_props(self):
        for p in self.props.values():
            g = self.get_state()[p["grip"]]
            sp, sR = _pose(self.d, p["site"])
            mid = p["mocap"]
            if p["held"] is None:
                p["armed"] = p["armed"] or g > GRIP_OPEN
                op = self.d.mocap_pos[mid]
                if p["armed"] and g < GRIP_CLOSE and np.linalg.norm((sp - op)[:2]) < GRASP_DIST:
                    oR = np.zeros(9)
                    mujoco.mju_quat2Mat(oR, self.d.mocap_quat[mid])
                    oR = oR.reshape(3, 3)
                    p["held"] = (sR.T @ (op - sp), sR.T @ oR)  # 그리퍼 좌표계 기준 물체 자세
                    p["armed"] = False
            elif g > GRIP_OPEN:  # 놓음: 그 자리 테이블 위에 세운다 (단순화)
                p["held"] = None
                p["armed"] = True
                self.d.mocap_pos[mid][2] = p["half"]
                self.d.mocap_quat[mid] = [1, 0, 0, 0]
            if p["held"] is not None:
                rp, rR = p["held"]
                self.d.mocap_pos[mid] = sp + sR @ rp
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (sR @ rR).ravel())
                self.d.mocap_quat[mid] = quat
        if self.props:
            mujoco.mj_forward(self.m, self.d)

    # ---------- 렌더링 ----------
    def render_frame(self, camera=None) -> np.ndarray:
        self.renderer.update_scene(self.d, camera=camera or self.camera)
        img = self.renderer.render()
        if self.inset is not None:
            img = _paste_inset(img, self.inset)
        if self.text:
            img = _draw_text(img, self.text)
        return img

    def _render(self):
        if self.writer is not None:  # 영상 없이 offscreen 이면 render_frame() 을 부를 때만 그린다
            self.writer.append_data(self.render_frame())
        if self.viewer is not None:
            self.viewer.sync()
            if self.realtime:
                now = time.perf_counter()
                if self._t_last is not None:
                    time.sleep(max(0.0, 1.0 / self.fps - (now - self._t_last)))
                self._t_last = time.perf_counter()

    def is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if (
            self.viewer is not None
        ):  # 뷰어 스레드가 끝나기 전에 인터프리터가 종료되면 core dump (Wayland/glfw)
            self.viewer.close()
            t0 = time.perf_counter()
            while self.viewer.is_running() and time.perf_counter() - t0 < 2.0:
                time.sleep(0.05)
            time.sleep(0.3)
            self.viewer = None


def _selftest():
    """단위 변환 왕복 시험 (범위 안 값은 그대로 돌아와야 함)."""
    m = build_model(props=False)
    rng = np.random.default_rng(0)
    lim = np.array([m.joint(n).range for n in NAMES])
    x_lo, x_hi = qpos_to_lerobot(lim[:, 0]), qpos_to_lerobot(lim[:, 1])
    x = rng.uniform(x_lo, x_hi, size=(1000, 12)).astype(np.float32)
    err = max(np.abs(qpos_to_lerobot(lerobot_to_qpos(v, m)) - v).max() for v in x)
    q = rng.uniform(lim[:, 0], lim[:, 1], size=(1000, 12))
    err2 = max(np.abs(lerobot_to_qpos(qpos_to_lerobot(v)) - v).max() for v in q)
    assert err < 1e-3 and err2 < 1e-5, (err, err2)
    assert np.allclose(lerobot_to_qpos([0] * 12)[GRIP], GRIP_LO)  # 그리퍼 0 = 닫힘 = 하한
    clipped = qpos_to_lerobot(lerobot_to_qpos(np.r_[np.zeros(10), 167.7, 0], m))[10]
    print(f"왕복 OK (LeRobot→rad→LeRobot 최대오차 {err:.2e}, rad→LeRobot→rad {err2:.2e})")
    print("LeRobot 단위 MJCF 범위:")
    for n, a, b in zip(NAMES, x_lo, x_hi):
        print(f"  {n:20s} {a:8.1f} .. {b:7.1f}")
    print(f"범위 밖 예: right_wrist_roll 167.7 → {clipped:.1f}")


if __name__ == "__main__":
    _selftest()
