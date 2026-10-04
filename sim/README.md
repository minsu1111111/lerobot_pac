# sim — 양팔 SO-101 MuJoCo 가짜 로봇

LeRobot `bi_so_follower`(SO-101 두 대)의 관절 명령을 MuJoCo 로 돌려 보는 폴더다.
이 폴더만 다른 프로젝트로 복사해도 동작한다 (바깥 모듈을 import 하지 않음).

- 관절 궤적 확인: 데이터셋 action/state, 모델 출력 npz 를 mp4·창·스틸로 재생
- 롤아웃 루프 시험: 실제 로봇 대신 `reset()/step()` 으로 12차원 관절 상태를 돌려주는 백엔드

## 한계 (꼭 읽을 것)

- **관절 수준 시뮬**이다. 렌더링 이미지는 정책 입력으로 쓰지 않는다 (카메라 영상은 실제/데이터셋 것을 써야 함).
  영상의 `inset`(오른쪽 위)은 사람이 보라고 붙이는 그림일 뿐이다.
- **액체 없음.** 컵·물통은 충돌 없는 mocap 모형이다. 그리퍼가 열렸다(>80) 닫힐 때(<60) 8 cm 안에 있으면
  붙고, 다시 열리면 그 자리 테이블 위에 똑바로 세운다 (단순화).
- 팔 충돌은 기본으로 꺼져 있다 (`contacts=False`; 접힌 자세 메시 겹침으로 튀는 것 방지).
- 두 팔 배치(`--spacing 0.45` m, `--yaw 30`°)는 **추정값**이다. 실측이 아니라 데이터셋 붓기 정점에서
  물통 입구가 컵 위에 오도록 격자 탐색 + top 카메라와 눈으로 비교해서 골랐다. 실제 리그를 재면 바꿀 것.
- 물리 모드 기본 `kp=17.8` (joints_properties.xml 의 옛 값). 실제 로봇 기록 상태와 가장 비슷했다.
  MJCF 원래 값 998.22 는 명령을 거의 지연 없이 따라가 실제보다 빠르다. 실제 모터 동역학 모델은 아니다.

## 설치

```bash
pip install -r requirements.txt
python fetch_so101.py          # 공식 SO-101 MJCF/STL (TheRobotStudio/SO-ARM100, Apache-2.0, 약 19MB)
python -m pytest -q tests      # 10개 시험, 렌더링 없음
```

공식 파일은 저장소에 넣지 않는다. 찾는 순서 (`sim_paths.so101_dir()`):

| 순서 | 위치 |
|---|---|
| 1 | 환경변수 `SO101_DIR` |
| 2 | `$UNITA_LOCAL/third_party/so101` (UNITA_LOCAL 이 있을 때) |
| 3 | `<sim>/third_party/so101` (이 폴더 단독으로 `fetch_so101.py` 를 돌린 경우, `.gitignore` 로 제외) |
| 4 | `<sim>/../../local/third_party/so101` (또는 ~/UNITA_PAC2026/local) |
| 5 | 없으면 `python fetch_so101.py` 를 돌리라는 오류 |

`fetch_so101.py` 기본 저장 위치: `UNITA_LOCAL` 이 있으면 `$UNITA_LOCAL/third_party/so101`,
`../../local` 이 있으면 그 아래, 아니면 `<sim>/third_party/so101`. `--dest DIR` 로 바꿀 수 있다 (그때는 `SO101_DIR=DIR`).

결과물(mp4·png·xml) 기본 폴더 (`sim_paths.outputs_dir()`):
`SIM_OUTPUTS` → `$UNITA_LOCAL/outputs/sim` → `../../local/outputs/sim` → `<sim>/outputs`.

## 단위 (LeRobot ↔ MJCF)

LeRobot `so_follower` (`use_degrees=True`) ↔ 공식 `so101_new_calib.xml` (라디안).
관절 순서는 데이터셋 `observation.state`/`action` 과 같다:
`left_shoulder_pan, left_shoulder_lift, left_elbow_flex, left_wrist_flex, left_wrist_roll, left_gripper, right_…` (12개, `.pos` 생략).
MJCF 관절·액추에이터 이름도 `left_shoulder_pan` 처럼 같다.

| 관절 | LeRobot 단위 | MJCF | 변환 | MJCF 범위 (LeRobot 단위) |
|---|---|---|---|---|
| shoulder_pan | deg, 0 = 보정 범위 중앙 | rad, 0 = 범위 중앙 | `rad = deg2rad(deg)` (부호 반전 없음) | −110.0 … 110.0 |
| shoulder_lift | deg | rad | 같음 | −100.0 … 100.0 |
| elbow_flex | deg | rad | 같음 | −96.8 … 96.8 |
| wrist_flex | deg | rad | 같음 | −95.0 … 95.0 |
| wrist_roll | deg | rad | 같음 | −157.2 … 162.8 |
| gripper | 0–100 (0 = 닫힘, 100 = 열림) | rad [−0.1745, 1.7453] | `rad = −0.1745 + x/100 · 1.9199` (선형) | 0 … 100 |

- `lerobot_to_qpos(x12, model)` 에 `model` 을 주면 MJCF 관절 범위로 **잘린다**
  (예: 데이터의 오른팔 wrist_roll 167.7° → 162.8°). 그래서 `step()` 이 돌려주는 상태도 범위 안이다.
- `qpos_to_lerobot(q12)` 는 float32 를 돌려준다.
- 부호(+1)는 영상·순기구학으로 확인했다.

## CLI

```bash
# 데이터셋 에피소드 재생 (기본 데이터셋: ~/.cache/huggingface/lerobot/UNITAmanipulation/bi_so101_pour_water_20260920_194823)
python play_trajectory.py --episode 0 --field action --stills auto
python play_trajectory.py --dataset /path/to/lerobot_dataset --episode 3 --camera top
# 물리 모드 + 추종오차 표 (시뮬 상태 vs 명령 vs 실제 기록 상태) + 창
python play_trajectory.py --episode 0 --physics --window
# 모델 출력 npz (키 ep{N}_demo, ep{N}_model, names): 모델(불투명) + 데모(반투명 분홍 겹침)
python play_trajectory.py --source npz --file actions.npz --episode 0 --which both
# 장면 XML 쓰기 / 뷰어로 보기
python bi_so101_scene.py --ghost --view
python bi_so101_scene.py --spacing 0.50 --yaw 35 --out scene.xml
# 단위 변환 자체 시험 + 범위 표
python mujoco_bi_so101.py
```

카메라: `front`(로봇 정면 위), `top`(데이터셋 top 카메라와 비슷), `side`.

## Python API — 다른 프로젝트에서 가짜 로봇으로 쓰기

```python
import sys; sys.path.insert(0, "path/to/sim")
from mujoco_bi_so101 import MujocoBiSO101, NAMES

sim = MujocoBiSO101(render="none", physics=True)   # "offscreen" + video_path="out.mp4" 이면 영상 저장
obs = sim.reset(state12)                            # LeRobot 단위 12차원 → 같은 단위 12차원 (float32)
for action12 in policy_actions:
    obs = sim.step(action12)                        # 위치 명령 → 1/fps 초 진행 → 시뮬 관절 상태
sim.close()
```

LeRobot `Robot` 처럼 dict 로 감싸는 예:

```python
KEYS = [f"{n}.pos" for n in NAMES]

class FakeBiSO101:
    def __init__(self, **kw):
        self.sim = MujocoBiSO101(**kw)
        self.state = None
    def connect(self, init_state12):
        self.state = self.sim.reset(init_state12)
    def get_observation(self):            # 카메라 이미지는 따로 (시뮬 렌더는 정책 입력용이 아님)
        return dict(zip(KEYS, map(float, self.state)))
    def send_action(self, action: dict):
        self.state = self.sim.step([action[k] for k in KEYS])
        return action
    def disconnect(self):
        self.sim.close()
```

주요 인자: `render` (`none` | `window` | `offscreen`), `video_path`, `fps=30`, `physics` (False = 자세 = 명령),
`spacing`, `yaw_deg`, `ghost` (반투명 두 번째 리그, `set_ghost(x12)`), `props`, `camera`, `kp`, `so101` (자산 폴더 직접 지정).
속성: `.text` (영상 왼쪽 위 글자, 여러 줄·한글 가능), `.inset` (오른쪽 위에 붙일 uint8 HWC 이미지), `.frames`.
`place_props_from_trajectory(traj)` 는 궤적을 순기구학으로 훑어 처음 잡는 위치에 컵·물통을 세운다.
영상 프레임 수 = `reset()` 1장 + `step()` 당 1장.

## 렌더링 백엔드

`MUJOCO_GL` 은 mujoco import 전에 정해야 한다. 비어 있으면 `DISPLAY` 가 있을 때 `glfw`, 없으면 `osmesa` 를 쓴다.

| MUJOCO_GL | 개발 머신 (Intel iGPU, 2026-10) |
|---|---|
| `glfw` | 동작, 640×480 약 70 fps. DISPLAY 필요 (offscreen 도 숨은 창으로 동작) |
| `egl` | 실패 (사용자가 render 그룹이 아님). GPU 서버에서는 보통 가장 빠름 |
| `osmesa` | 동작하지만 약 2.5 fps. 디스플레이 없는 곳에서만 |

`render="none"` 이면 OpenGL 을 쓰지 않는다 (시험·헤드리스 롤아웃).

## 파일

| 파일 | 내용 |
|---|---|
| `bi_so101_scene.py` | MjSpec 으로 공식 팔 두 개를 붙인 장면 (`left_`/`right_` 접두어, 소품, ghost 리그) |
| `mujoco_bi_so101.py` | `MujocoBiSO101`, `lerobot_to_qpos`, `qpos_to_lerobot` |
| `play_trajectory.py` | 데이터셋/npz 재생 → mp4·창·스틸, 물리 추종오차 |
| `fetch_so101.py` | 공식 SO-101 파일 받기 |
| `sim_paths.py` | 자산·결과물 경로 규칙 |
| `tests/` | pytest (단위 변환 왕복, 관절 이름, reset/step, 운동학 모드) |
