"""로봇 백엔드 3종. 인터페이스는 lerobot Robot 과 같은 이름만 쓴다.

  connect() / get_observation() -> dict / send_action(dict) -> dict / disconnect()
  observation_features / action_features (BiSOFollower 와 같은 형식: 관절 float, 카메라 (H, W, 3))

관측 dict 키는 실제 BiSOFollower.get_observation() 과 같다 (bi_so_follower.py 확인):
  관절   "left_shoulder_pan.pos" … "left_gripper.pos", "right_shoulder_pan.pos" … "right_gripper.pos"
         (SOFollower 가 "{motor}.pos" 를 만들고 BiSOFollower 가 "left_"/"right_" 를 붙임)
  카메라 "top"        : BiSOFollowerConfig.cameras (최상위 카메라, 접두사 없음)
         "left_wrist" : left_arm_config.cameras["wrist"]  → "left_" 접두사
         "right_wrist": right_arm_config.cameras["wrist"] → "right_" 접두사
  이미지 = OpenCVCamera.read_latest() 와 같은 uint8 HWC RGB.

백엔드
  real          실제 BiSOFollower. 이 머신에서는 시험 못 함 (하드웨어 없음) — 소스대로만 맞춤.
  replay        데이터셋 에피소드 한 개를 그대로 흘려보냄 (open-loop: 명령은 기록만, 관측에 반영 안 됨).
  replay+mujoco 이미지는 데이터셋, 관절 state 는 MuJoCo 가 모델 명령을 따라 움직인 값 (관절만 closed-loop).
                [근사] 이미지는 명령에 반응하지 않으므로 '팔이 따라 움직이는' 상황의 근사일 뿐이다.
"""

import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASET, GITHUB  # noqa: E402

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_KEYS = [f"{s}_{m}.pos" for s in ("left", "right") for m in MOTORS]  # 데이터셋 action/state 순서
CAM_SHAPES = {"top": (480, 640, 3), "left_wrist": (240, 320, 3), "right_wrist": (240, 320, 3)}


def calib_dir() -> Path:
    """Robot.__init__ 과 같은 규칙: HF_LEROBOT_CALIBRATION(환경변수로 변경 가능)/robots/<SOFollower.name="so_follower">."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    return HF_LEROBOT_CALIBRATION / ROBOTS / "so_follower"


def _features():
    return {**dict.fromkeys(JOINT_KEYS, float), **CAM_SHAPES}


# --------------------------------------------------------------------------- #
# real
# --------------------------------------------------------------------------- #
def calib_paths(robot_id: str) -> tuple[Path, Path]:
    """BiSOFollower 는 팔별 id 를 f"{id}_left" / f"{id}_right" 로 만든다 (bi_so_follower.py __init__).
    calibration_dir 를 안 주면 Robot.__init__ 이 HF_LEROBOT_CALIBRATION/robots/<SOFollower.name>/ 를 쓴다."""
    d = calib_dir()
    return d / f"{robot_id}_left.json", d / f"{robot_id}_right.json"


def install_calib(robot_id: str, left_src: str, right_src: str):
    """rig_config 의 팔 하나짜리 캘리브레이션(follower1.json 등)을 양팔용 이름으로 복사한다.
    src 는 id(예: follower1) 또는 json 경로. 대상이 이미 있고 내용이 다르면 덮어쓰지 않고 멈춘다."""
    for src, dst in zip((left_src, right_src), calib_paths(robot_id)):
        p = Path(src) if src.endswith(".json") else calib_dir() / f"{src}.json"
        if not p.is_file():
            raise FileNotFoundError(f"캘리브레이션 원본 없음: {p} (rig_config/install.sh 먼저)")
        if dst.is_file():
            if dst.read_bytes() != p.read_bytes():
                raise FileExistsError(f"{dst} 가 이미 있고 {p} 와 다릅니다. 확인 후 직접 지우세요.")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        print(f"[calib] {p.name} → {dst}")


class RealRobot:
    """lerobot BiSOFollower 를 그대로 감싼다 (lerobot-rollout 의 make_robot_from_config 와 같은 객체).

    [미검증] 하드웨어가 없어 이 머신에서 connect 이후는 돌려보지 못했다. 설정 객체 생성까지만 확인.
    """

    name = robot_type = "bi_so_follower"

    def __init__(
        self,
        robot_id: str,
        left_port: str,
        right_port: str,
        top_cam: str,
        left_cam: str,
        right_cam: str,
        max_relative_target: float | None = None,
        keep_torque: bool = False,
        fourcc: str | None = None,
    ):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
        from lerobot.robots.so_follower import SOFollowerConfig

        def cam(path, shape):
            h, w, _ = shape
            src = int(path) if str(path).isdigit() else Path(path)
            return OpenCVCameraConfig(index_or_path=src, fps=30, width=w, height=h, fourcc=fourcc)

        # ensure_safe_goal_position 은 float/dict 만 받는다 (int 면 TypeError)
        mrt = None if max_relative_target is None else float(max_relative_target)
        arm = dict(max_relative_target=mrt, disable_torque_on_disconnect=not keep_torque, use_degrees=True)
        cfg = BiSOFollowerConfig(
            id=robot_id,
            left_arm_config=SOFollowerConfig(
                port=left_port, cameras={"wrist": cam(left_cam, CAM_SHAPES["left_wrist"])}, **arm
            ),
            right_arm_config=SOFollowerConfig(
                port=right_port, cameras={"wrist": cam(right_cam, CAM_SHAPES["right_wrist"])}, **arm
            ),
            cameras={"top": cam(top_cam, CAM_SHAPES["top"])},
        )
        missing = [str(p) for p in calib_paths(robot_id) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "캘리브레이션 파일 없음 → connect 때 대화형 캘리브레이션이 시작되므로 중단합니다.\n  "
                + "\n  ".join(missing)
                + "\n  해결: --calib-left follower? --calib-right follower? 로 복사 (어느 팔이 follower1/2 인지 확인)"
            )
        self.robot = BiSOFollower(cfg)
        self.initial_position: dict | None = None

    @property
    def observation_features(self):
        return self.robot.observation_features

    @property
    def action_features(self):
        return self.robot.action_features

    @property
    def is_connected(self):
        return self.robot.is_connected

    def connect(self):
        # 캘리브레이션 파일과 모터 값이 다르면 SOFollower.calibrate() 가 input() 으로 묻는다 (Enter = 파일 사용)
        self.robot.connect()
        obs = self.robot.get_observation()
        self.initial_position = {k: obs[k] for k in JOINT_KEYS}

    def get_observation(self):
        return self.robot.get_observation()

    def send_action(self, action):
        return self.robot.send_action(action)

    def return_to_initial(self, duration_s=3.0, fps=50):
        """lerobot RolloutStrategy._return_to_initial_position 과 같은 선형 보간 복귀."""
        if not self.initial_position or not self.robot.is_connected:
            return
        cur = self.robot.get_observation()
        n = max(int(duration_s * fps), 1)
        for i in range(1, n + 1):
            t = i / n
            self.robot.send_action({k: cur[k] * (1 - t) + v * t for k, v in self.initial_position.items()})
            time.sleep(1 / fps)

    def disconnect(self):
        if self.robot.is_connected:
            self.robot.disconnect()


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
class DatasetReplayRobot:
    """데이터셋 에피소드 한 개를 30fps 로봇처럼 흘려보낸다 (open-loop).

    clock="step": get_observation() 한 번에 한 프레임씩 (결정적, 기본)
    clock="wall": connect 이후 경과 시간으로 프레임을 고른다 (루프가 밀리면 실제 카메라처럼 프레임을 건너뜀)
    영상은 백그라운드 스레드가 블록(60프레임) 단위로 미리 디코딩한다 (3캠 합계 ≈ 40~120 fps, 부하에 따라).
    메모리: 앞으로 LOOKAHEAD 프레임까지만 디코딩하고 지나간 프레임은 버린다 (프레임당 1.4MB).
    restart() 는 디코딩을 처음부터 다시 시작한다.

    hold_last=False: 마지막 프레임에서 finished=True → pour_rollout 이 stop:replay_end 로 끝낸다 (기본, 예전 동작).
    hold_last=True : 에피소드가 끝나도 finished 를 세우지 않고 마지막 카메라 프레임·state 를 계속 준다
                     → 루프는 done(post-done) 또는 timeout 으로 끝난다. episode_ended 로 끝났는지만 알 수 있다.
    """

    name = robot_type = "bi_so_follower"
    BLOCK = 60
    LOOKAHEAD = 300  # 10s 분량 ≈ 420MB

    def __init__(
        self,
        episode: int,
        dataset: Path = DATASET,
        clock: str = "step",
        fps: int = 30,
        hold_last: bool = False,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self.meta = LeRobotDatasetMetadata(dataset.name, root=dataset)
        self.root, self.episode, self.clock, self.fps = dataset, episode, clock, fps
        self.hold_last = hold_last
        ep = self.meta.episodes[episode]
        self.ep = ep
        self.length = int(ep["length"])
        names = self.meta.features["observation.state"]["names"]
        assert names == JOINT_KEYS, names
        self.state, self.demo_action = self._load_table()
        self._cv = threading.Condition()
        self._gen = 0  # restart 마다 증가 → 옛 디코딩 스레드 종료
        self._t0 = None
        self.decode_wait_s = 0.0
        self._reset_buffer()

    def _reset_buffer(self):
        self.frames: dict[int, dict] = {}
        self.ready = 0
        self.idx = -1
        self.finished = False  # pour_rollout 이 보고 stop:replay_end (hold_last 면 항상 False)
        self.episode_ended = False  # 에피소드 마지막 프레임에 도달했는지 (hold_last 와 무관)
        self.sent: list[np.ndarray] = []

    def _load_table(self):
        import pandas as pd

        df = pd.read_parquet(
            self.root / self.meta.get_data_file_path(self.episode),
            columns=["episode_index", "frame_index", "observation.state", "action"],
        )
        df = df[df.episode_index == self.episode].sort_values("frame_index")
        assert len(df) == self.length
        return (
            np.stack(df["observation.state"].values).astype(np.float32),
            np.stack(df["action"].values).astype(np.float32),
        )

    def _decode(self, gen: int):
        from lerobot.datasets.video_utils import decode_video_frames

        for lo in range(0, self.length, self.BLOCK):
            with self._cv:  # 너무 앞서가지 않게 대기
                while self._gen == gen and lo - max(self.idx, 0) > self.LOOKAHEAD:
                    self._cv.wait(0.5)
                if self._gen != gen:
                    return
            hi = min(lo + self.BLOCK, self.length)
            block = {}
            for c in CAM_SHAPES:
                key = f"observation.images.{c}"
                f0 = self.ep[f"videos/{key}/from_timestamp"]
                path = self.root / self.meta.get_video_file_path(self.episode, key)
                ts = [f0 + i / self.fps for i in range(lo, hi)]
                fr = decode_video_frames(path, ts, 1e-4, None, return_uint8=True)  # (N, C, H, W)
                block[c] = fr.permute(0, 2, 3, 1).numpy()
            with self._cv:
                if self._gen != gen:
                    return
                for i in range(lo, hi):
                    self.frames[i] = {c: np.ascontiguousarray(block[c][i - lo]) for c in CAM_SHAPES}
                self.ready = hi
                self._cv.notify_all()

    observation_features = property(lambda self: _features())
    action_features = property(lambda self: dict.fromkeys(JOINT_KEYS, float))
    is_connected = property(lambda self: self._t0 is not None)

    def _start_decoder(self):
        with self._cv:
            self._gen += 1
            self._reset_buffer()
            self._cv.notify_all()
        t = threading.Thread(target=self._decode, args=(self._gen,), name="replay-decode", daemon=True)
        t.start()
        self._threads = [x for x in getattr(self, "_threads", []) if x.is_alive()] + [t]

    def connect(self):
        self._start_decoder()
        self._t0 = time.perf_counter()

    def restart(self):
        """다음 붓기 시작 때 에피소드를 처음부터 다시 (IDLE → pour 반복 시험용)."""
        if self.idx >= 0:  # 이미 진행했으면 처음부터 다시 디코딩
            self._start_decoder()
        self._t0 = time.perf_counter()

    def _next_index(self):
        if self.clock == "wall":
            i = int((time.perf_counter() - self._t0) * self.fps)
        else:
            i = self.idx + 1
        if i >= self.length - 1:
            self.episode_ended = True
            self.finished = not self.hold_last
        return min(i, self.length - 1)

    def _frame(self, i):
        t = time.perf_counter()
        with self._cv:
            while self.ready <= i:
                self._cv.wait(1.0)
            out = self.frames[i]
            for j in [j for j in self.frames if j < i]:  # 지나간 프레임 버림
                del self.frames[j]
            self._cv.notify_all()
        self.decode_wait_s += time.perf_counter() - t
        return out

    def get_observation(self):
        i = self._next_index()
        self.idx = i
        obs = {k: float(v) for k, v in zip(JOINT_KEYS, self.state[i])}
        obs.update(self._frame(i))
        return obs

    def send_action(self, action):
        self.sent.append(np.array([action[k] for k in JOINT_KEYS], dtype=np.float32))
        return action

    def disconnect(self):
        with self._cv:
            self._gen += 1
            self._cv.notify_all()
        # 디코더 스레드가 영상 디코딩(C++) 도중에 인터프리터가 끝나면 abort 가 나므로 끝날 때까지 기다린다
        for t in getattr(self, "_threads", []):
            t.join(timeout=10.0)
        self._t0 = None


class ReplayMujocoRobot(DatasetReplayRobot):
    """이미지 = 데이터셋, 관절 state = MuJoCo (모델 명령을 따라감). [근사] 이미지는 명령에 반응하지 않는다.

    hold_last=True (기본): 데이터셋 영상이 끝나도 마지막 카메라 프레임을 계속 주고 시뮬은 모델 명령대로 계속 움직인다
    → DONE 후 post-done 구간(물통 내려놓기)까지 영상에 담긴다. 시뮬 영상 프레임 = reset 1장 + step 당 1장 (변함 없음)."""

    def __init__(
        self,
        episode: int,
        dataset: Path = DATASET,
        clock: str = "step",
        fps: int = 30,
        render: str = "none",
        video_path: str | None = None,
        hold_last: bool = True,
    ):
        super().__init__(episode, dataset, clock, fps, hold_last=hold_last)
        sys.path.insert(0, str(GITHUB / "sim"))
        from mujoco_bi_so101 import MujocoBiSO101  # 다른 담당 모듈 (sim/)

        self.sim = MujocoBiSO101(render=render, video_path=video_path, fps=fps, physics=True)
        self.sim_state = None

    def restart(self):
        super().restart()
        self.sim_state = None

    def get_observation(self):
        i = self._next_index()
        self.idx = i
        if self.sim_state is None:  # 붓기 시작마다 데이터셋 첫 프레임 자세로 리셋
            self.sim_state = np.asarray(self.sim.reset(self.state[i]), dtype=np.float32)
            self.sim.place_props_from_trajectory(self.demo_action)  # 컵·물통을 시연에서 잡은 자리에 세움
        obs = {k: float(v) for k, v in zip(JOINT_KEYS, self.sim_state)}
        obs.update(self._frame(i))
        self.sim.inset = obs.get("top")  # 영상 오른쪽 위: 정책이 보는 top 카메라
        return obs

    def set_overlay(self, text: str):
        """시뮬 영상 왼쪽 위 글자 (감지 상태·TTS 자막). 다음 프레임부터 찍힌다."""
        self.sim.text = text

    def send_action(self, action):
        a = np.array([action[k] for k in JOINT_KEYS], dtype=np.float32)
        self.sent.append(a)
        self.sim_state = np.asarray(self.sim.step(a), dtype=np.float32)
        return action

    def disconnect(self):
        super().disconnect()
        self.sim.close()
