# UNITA PAC 2026 — 말하면 따라주는 손

> 이 저장소는 [Unita_lerobot](https://github.com/minsu1111111/Unita_lerobot) 의 `pac2026/` 폴더와 같은 내용이다.
> 문서에 나오는 `pac2026/...` 경로는 이 저장소 최상위로 읽으면 된다.
> 실행에는 LeRobot 0.6.1 팀 포크(Unita_lerobot)를 editable 설치한 환경이 필요하다 (`pip install -e ~/lerobot`).

PAC 2026 피지컬 AI 챌린지 (분과① 자율주제, UNITA Manipulation · 인천대).
음성 명령 → 양팔 SO-101(LeRobot `bi_so_follower`)이 ACT 정책으로 왼팔은 컵을 잡고 오른팔은 물통으로 물을 따름
→ 관절 상태로 완료를 감지해 TTS 로 안내한다.

- 데이터셋: `UNITAmanipulation/bi_so101_pour_water_20260920_194823` (100 에피소드, 30 fps, 카메라 3대, 12차원)
- 모델: `UNITAmanipulation/act_pour_water_100` (ACT, chunk 100)
- LeRobot v0.6.1 소스 기준으로 맞춤 (API 는 설치된 소스를 읽고 확인함)

## 폴더

`tts/` `stt/` `sim/` 은 **서로 독립**이다. 폴더 하나만 다른 프로젝트에 복사해도 동작하고,
각자 README · requirements.txt · tests 가 있다. 나머지는 이 셋과 공용 코드를 묶는 쪽이다.

| 폴더 | 내용 | 독립 |
|---|---|---|
| [`tts/`](tts/) | 고정 문구 wav 를 비차단 재생 (제어 루프를 막지 않음, 스피커 없으면 자막만) | ✅ |
| [`stt/`](stt/) | 말소리 자동 감지(VAD) 또는 Enter 녹음 → Whisper(ko) → `따라줘`/`정지` 명령 매칭, 키보드 대체 입력, 인식률 시험·녹음 도구 | ✅ |
| [`sim/`](sim/) | 양팔 SO-101 MuJoCo: 궤적 재생 영상, 롤아웃용 가짜 관절 로봇 | ✅ |
| [`rollout/`](rollout/) | 데모 롤아웃 루프 (`pour_rollout.py`): 감지기 + TTS + STT + 수동 정지, 백엔드 real / replay / replay+mujoco | 묶는 쪽 |
| [`analysis/`](analysis/) | 관절 분석 → 완료 감지 임계값 (`--install` 로 `rollout/thresholds.json` 갱신) | |
| [`validate/`](validate/) | 오프라인(open-loop) 모델 검증: 데이터셋 프레임 → 모델 action → 감지기 | |
| [`train/`](train/) | 데이터 수집(`record.sh`, lerobot-record 양팔), 학습(`train.sh`, 이어서 학습/처음부터), 체크포인트 받아오기·검증 절차 | |
| [`tools/`](tools/) | `offline_check.py`(인터넷 차단 점검), `camera_check.py`(카메라·초점·수위 확인), `tts_loop_timing.py`, `mux_tts_audio.py`(시뮬 영상에 TTS 소리) | |
| `pour_detector.py` | 완료 감지기 (오른팔 wrist_roll 히스테리시스 + 타임아웃) | |
| `paths.py` | 경로 모음 (`UNITA_LOCAL` = 깃에 안 올리는 결과물·외부 파일, 기본 `~/UNITA_PAC2026/local`) | |
| `run_demo.sh` | 단일 실행기 (오프라인 점검 → 롤아웃) | |

저장소 밖 `~/UNITA_PAC2026/local/` (환경변수 `UNITA_LOCAL` 로 변경 가능, 깃 제외): `outputs/`(분석·영상·로그), `third_party/so101`(공식 SO-101 MJCF, `sim/fetch_so101.py`),
`robot.env`(PC 별 포트·카메라, 예시는 `rollout/robot.env.example`).

대회 전 준비 순서는 [`PREP_CHECKLIST.md`](PREP_CHECKLIST.md) (GPU 확인, 카메라 확인, 녹화·학습 리허설, STT 시험).

## 실행

```bash
PY=~/miniconda3/envs/lerobot/bin/python      # conda run 대신 env 의 python 직접 사용

bash run_demo.sh --replay                    # 하드웨어 없이: 데이터셋 재생 (open-loop)
bash run_demo.sh --sim                       # 하드웨어 없이: 데이터셋 영상 + MuJoCo 관절 (근사 closed-loop)
bash run_demo.sh                             # 실제 로봇 (~/UNITA_PAC2026/local/robot.env 필요)

$PY tools/offline_check.py --block-network   # 현장 전: 인터넷 없이 전부 로드되는지
```

조작: 대기 중 "물 따라줘" 라고 말하면 시작 (Enter 불필요, `p`+Enter 도 됨. 시끄러우면 `--stt-mode enter` 로 Enter→말하기→Enter)
/ 붓는 중 Space·`s` = 즉시 정지, `q` = 종료, Ctrl+C = 안전 종료. 안내 음성이 끝난 뒤에만 듣는다.
완료 감지 즉시 "다 따랐습니다" 안내, 12초 뒤 정지 후 초기 자세 복귀. 명령을 못 알아들으면 "다시 말씀해 주세요".

시뮬 데모 영상 (TTS 소리 포함):

```bash
$PY rollout/pour_rollout.py --backend replay+mujoco --episode 25 --fast --auto-start --no-stt --no-tts \
    --sim-render offscreen --sim-video ~/UNITA_PAC2026/local/outputs/sim/rollout_ep25_sim.mp4
```

## 주요 결정과 측정값

- **완료 감지 = 관절 상태.** 오른팔 `wrist_roll` 이 기준선 + 78.9° 를 5프레임 넘으면 붓는 중, + 39.4° 안으로 5프레임 들어오면 완료.
  타임아웃 63 s. 데이터셋 100/100, 모델 출력(open-loop) 100/100 감지. 비전 수위 판정은 하지 않음.
- **청크 경계 튐:** ACT 가 100스텝마다 새 청크를 낼 때 open-loop 에서 팔 관절이 최대 70°/프레임 튐
  → `--max-step-deg 10` 기본 (시연 데이터 최대 ≈10°/프레임, 정상 동작에선 몇 프레임만 걸림). MuJoCo closed-loop 에선 7°.
- **오프라인 현장:** ResNet18 ImageNet 가중치 다운로드를 막으려고 `pretrained_backbone_weights=None` (체크포인트가 덮어써서 출력 동일).
- **STT:** Whisper small, 말소리 자동 감지(기본, `--vad-level` 로 민감도), 프롬프트 `"물 따라줘. 정지."`. 합성 음성 122개 98.4%, stop 누락 0, 오작동 pour 0 (실제 목소리 시험 전).

## 하드웨어 없이 검증한 범위

데이터셋 재생과 MuJoCo 관절 시뮬까지다. 실제 로봇 백엔드는 lerobot 소스에 맞춰 작성했지만 실행해 보지 못했다.
첫 실물 시험에서 확인할 것: 완료 감지 임계값(손목 최대 각도), 청크 경계 튐, 정지 키, 현장 소음에서 STT, GPU(torch CUDA) 동작.
