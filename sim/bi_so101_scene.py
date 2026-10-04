"""양팔 SO-101 MuJoCo 장면을 만든다 (공식 so101_new_calib.xml 두 개를 MjSpec 으로 붙임).

좌표: 팔 정면 = +x, 왼팔 = +y 쪽, 오른팔 = -y 쪽 (로봇 뒤에서 앞을 볼 때 기준).
  - 팔 받침(base) 바닥 = 테이블 면 z=0
  - 이름 접두어: left_ / right_ (관절·액추에이터 이름이 LeRobot 키와 같음: left_shoulder_pan ...)
  - ghost=True 이면 반투명 두 번째 리그(ghost_left_ / ghost_right_)를 같은 자리에 겹친다 (데모 vs 모델 비교용)
  - props=True 이면 컵·물통 모형(mocap, 충돌 없음, 액체 없음)을 넣는다

공식 파일(sim_paths.so101_dir(): SO101_DIR → sim/third_party/so101 → ../local/third_party/so101)은 수정하지 않는다.
meshdir 은 절대경로로 바꿔서 붙이므로 --out 으로 쓴 XML 은 (그 컴퓨터 안에서는) 어디서든 열린다.

  python bi_so101_scene.py                 # <outputs>/bi_so101_scene.xml (sim_paths.outputs_dir())
  python bi_so101_scene.py --ghost --view  # 인터랙티브 창으로 보기
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_paths import ARM_XML_NAME, outputs_dir, so101_dir  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SIDES = ["left", "right"]

# 소품 크기 (m). 실제 영상 기준 대략값: 물통 = 투명 유리병, 컵 = 남색 텀블러
BOTTLE = dict(radius=0.035, half=0.09, rgba=[0.75, 0.9, 1.0, 0.45])
CUP = dict(radius=0.040, half=0.06, rgba=[0.10, 0.15, 0.55, 1.0])


def _arm_spec(so101: Path, alpha: float = 1.0, tint=None, contacts: bool = False) -> mujoco.MjSpec:
    s = mujoco.MjSpec.from_file(str(so101 / ARM_XML_NAME))
    s.meshdir = str(so101 / "assets")
    for g in s.geoms:
        if not contacts:
            g.contype = g.conaffinity = 0
    for mat in s.materials:
        rgba = np.array(mat.rgba, float)
        if tint is not None:
            rgba[:3] = 0.5 * rgba[:3] + 0.5 * np.asarray(tint)
        rgba[3] = alpha
        mat.rgba = rgba
    return s


def build_spec(
    spacing: float = 0.45,
    x_offset: float = 0.0,
    yaw_deg: float = 30.0,
    ghost: bool = False,
    props: bool = True,
    contacts: bool = False,
    timestep: float = 1 / 600,
    kp: float | None = None,
    kv: float | None = None,
    so101: str | Path | None = None,
) -> mujoco.MjSpec:
    """spacing: 두 팔 받침 사이 거리(m). yaw_deg: 두 팔을 안쪽(+)으로 돌리는 각도.
    기본값 0.45 m / 30° 는 실측이 아니라 추정: 데이터셋 8개 에피소드의 붓기 정점에서 물통 입구가 컵 위에 오도록
    격자 탐색(0.40/25°, 0.50/35° 도 비슷) + top 카메라 영상과 눈으로 비교. 실제 리그를 재면 바꿀 것.
    contacts=False: 팔 충돌 끔 (관절 수준 가짜 로봇 용도. 접힌 자세의 메시 겹침으로 튀는 것 방지).
    kp/kv: 위치 액추에이터 게인 덮어쓰기. None 이면 so101_new_calib.xml 값 (kp=998.22, kv=2.731, 토크 ±3.35 N·m).
           joints_properties.xml 의 옛 값은 kp=17.8, kv=0.
    so101: SO-101 자산 폴더 (None = sim_paths.so101_dir() 규칙)."""
    so101 = Path(so101) if so101 is not None else so101_dir()
    w = mujoco.MjSpec()
    w.modelname = "bi_so101"
    w.meshdir = str(so101 / "assets")  # to_xml 로 쓴 파일이 어디서든 열리도록 절대경로
    w.option.timestep = timestep
    w.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    w.visual.headlight.ambient = [0.35, 0.35, 0.35]
    w.visual.global_.offwidth, w.visual.global_.offheight = 1280, 960

    w.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.35, 0.45, 0.55],
        rgb2=[0.05, 0.05, 0.08],
        width=256,
        height=1536,
    )
    w.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.82, 0.78, 0.70],
        rgb2=[0.76, 0.72, 0.64],
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.6, 0.58, 0.52],
        width=200,
        height=200,
    )
    tex = mujoco.mjtTextureRole.mjTEXROLE_RGB
    mat = w.add_material(name="table", texuniform=True, texrepeat=[10, 10], reflectance=0.0)
    mat.textures[tex] = "grid"
    wb = w.worldbody
    wb.add_light(
        pos=[0.3, 0, 1.5],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=True,
        diffuse=[0.6, 0.6, 0.6],
    )
    # 테이블 면 = 바닥 평면 (10 cm 격자: texrepeat 10 / 1 m)
    wb.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.05],
        pos=[0.25, 0, 0],
        material="table",
    )

    rigs = [("", 1.0, None)] + ([("ghost_", 0.35, [1.0, 0.1, 0.6])] if ghost else [])
    for pre, alpha, tint in rigs:
        for side, sgn in (("left", 1), ("right", -1)):
            yaw = np.deg2rad(-sgn * yaw_deg)
            f = wb.add_frame(
                pos=[x_offset, sgn * spacing / 2, 0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            )
            w.attach(_arm_spec(so101, alpha, tint, contacts and not pre), prefix=f"{pre}{side}_", frame=f)

    # 고정 카메라: front = 로봇 정면 위에서, top = 데이터셋 top 카메라와 비슷한 방향(팔이 화면 위쪽)
    wb.add_camera(name="front", pos=[0.95, 0.0, 0.62], xyaxes=[0, 1, 0, -0.55, 0, 0.83])
    wb.add_camera(name="top", pos=[0.30, 0.0, 1.05], xyaxes=[0, 1, 0, -1, 0, 0])
    wb.add_camera(name="side", pos=[0.25, -0.95, 0.45], xyaxes=[1, 0, 0, 0, 0.4, 0.92])

    for act in w.actuators:
        if kp is not None:
            act.gainprm[0], act.biasprm[1] = kp, -kp
        if kv is not None:
            act.biasprm[2] = -kv

    if props:  # mocap 소품: 위치는 MujocoBiSO101 이 그리퍼 상태로 갱신
        for name, p, y in (("bottle", BOTTLE, -spacing / 2), ("cup", CUP, spacing / 2)):
            b = wb.add_body(name=name, mocap=True, pos=[0.25, y, p["half"]])
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[p["radius"], p["half"], 0],
                rgba=p["rgba"],
                contype=0,
                conaffinity=0,
            )
            if name == "bottle":  # 병목 (기울기 방향이 보이도록)
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.015, 0.02, 0],
                    pos=[0, 0, p["half"] + 0.02],
                    rgba=[0.9, 0.9, 0.95, 0.8],
                    contype=0,
                    conaffinity=0,
                )
    return w


def build_model(**kw) -> mujoco.MjModel:
    return build_spec(**kw).compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.45)
    ap.add_argument("--yaw", type=float, default=30.0)
    ap.add_argument("--ghost", action="store_true")
    ap.add_argument("--no-props", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="기본: <outputs>/bi_so101_scene.xml")
    ap.add_argument("--view", action="store_true", help="mujoco.viewer 로 열기")
    a = ap.parse_args()
    spec = build_spec(spacing=a.spacing, yaw_deg=a.yaw, ghost=a.ghost, props=not a.no_props)
    m = spec.compile()
    a.out = a.out or outputs_dir() / "bi_so101_scene.xml"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(spec.to_xml())
    print(f"nq={m.nq} nu={m.nu} → {a.out}")
    if a.view:
        import mujoco.viewer

        mujoco.viewer.launch(m)


if __name__ == "__main__":
    main()
