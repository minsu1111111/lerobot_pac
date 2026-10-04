import numpy as np
import pytest

from mujoco_bi_so101 import GRIP, GRIP_HI, GRIP_LO, NAMES, MujocoBiSO101, lerobot_to_qpos, qpos_to_lerobot

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _limits(model):
    return np.array([model.joint(n).range for n in NAMES])


# ---------- 단위 변환 (모델 없이) ----------
def test_names_order():
    assert NAMES == [f"{s}_{j}" for s in ("left", "right") for j in JOINTS]


def test_conversion_known_values():
    x = np.zeros(12)
    x[0], x[6] = 90.0, -45.0
    x[GRIP] = [0.0, 100.0]
    q = lerobot_to_qpos(x)
    assert np.isclose(q[0], np.pi / 2) and np.isclose(q[6], -np.pi / 4)
    assert np.isclose(q[5], GRIP_LO) and np.isclose(q[11], GRIP_HI)  # 0 = 닫힘 = 하한, 100 = 열림 = 상한
    assert np.allclose(lerobot_to_qpos(np.full(12, 50.0))[GRIP], (GRIP_LO + GRIP_HI) / 2)


def test_round_trip_unclipped():
    rng = np.random.default_rng(0)
    x = rng.uniform(-150, 150, size=(200, 12))
    x[:, GRIP] = rng.uniform(0, 100, size=(200, 2))
    for v in x:
        back = qpos_to_lerobot(lerobot_to_qpos(v))
        assert back.dtype == np.float32 and back.shape == (12,)
        assert np.abs(back - v).max() < 1e-3


# ---------- 모델 필요 ----------
def test_build_model_joint_names(model):
    names = {model.joint(i).name for i in range(model.njnt)}
    assert set(NAMES) <= names
    for n in NAMES:  # 위치 액추에이터 이름 = 관절 이름
        assert model.actuator(n).id >= 0
    assert model.nu == 12


def test_round_trip_within_limits(model):
    lim = _limits(model)
    rng = np.random.default_rng(1)
    q = rng.uniform(lim[:, 0], lim[:, 1], size=(200, 12))
    for v in q:
        assert np.abs(lerobot_to_qpos(qpos_to_lerobot(v), model) - v).max() < 1e-5


def test_clipping(model):
    lim = _limits(model)
    x = np.zeros(12)
    x[NAMES.index("right_wrist_roll")] = 167.7  # 데이터에 있는 값, MJCF 상한(약 162.8°) 초과
    x[GRIP] = [-20.0, 130.0]
    q = lerobot_to_qpos(x, model)
    assert np.all(q >= lim[:, 0] - 1e-12) and np.all(q <= lim[:, 1] + 1e-12)
    assert np.isclose(q[NAMES.index("right_wrist_roll")], lim[NAMES.index("right_wrist_roll"), 1])


@pytest.mark.parametrize("physics", [True, False])
def test_reset_step_shapes_and_range(so101, physics):
    sim = MujocoBiSO101(render="none", physics=physics, so101=so101)
    try:
        lim = _limits(sim.m)
        lo, hi = qpos_to_lerobot(lim[:, 0]), qpos_to_lerobot(lim[:, 1])
        s0 = np.zeros(12, np.float32)
        s0[GRIP] = 10.0
        obs = sim.reset(s0)
        assert obs.shape == (12,) and obs.dtype == np.float32
        assert np.allclose(obs, s0, atol=1e-3)
        cmd = s0.copy()
        cmd[[1, 7]] = [-30.0, -30.0]
        for _ in range(30):
            obs = sim.step(cmd)
            assert obs.shape == (12,) and np.all(np.isfinite(obs))
            assert np.all(obs >= lo - 1.0) and np.all(obs <= hi + 1.0)
        assert sim.frames == 30
        if physics:  # 1초 뒤에는 명령 쪽으로 움직였어야 함
            assert obs[1] < -5.0 and obs[7] < -5.0
    finally:
        sim.close()


def test_kinematic_equals_command(so101):
    sim = MujocoBiSO101(render="none", physics=False, props=False, so101=so101)
    try:
        lim = _limits(sim.m)
        rng = np.random.default_rng(2)
        sim.reset(np.zeros(12))
        for q in rng.uniform(lim[:, 0], lim[:, 1], size=(20, 12)):
            cmd = qpos_to_lerobot(q)
            assert np.allclose(sim.step(cmd), cmd, atol=1e-3)
    finally:
        sim.close()


def test_props_follow_gripper(so101):
    """그리퍼를 열었다 닫으면 가까운 소품이 붙는다 (place_props_from_trajectory 로 그 자리에 세움)."""
    sim = MujocoBiSO101(render="none", physics=False, so101=so101)
    try:
        open_ = np.zeros(12)
        open_[GRIP] = 100.0
        closed = np.zeros(12)
        sim.place_props_from_trajectory(np.stack([open_, closed]))
        sim.reset(open_)
        sim.step(closed)
        assert all(p["held"] is not None for p in sim.props.values())
    finally:
        sim.close()
