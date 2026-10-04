# 대회 전 준비 체크리스트 (10/5 ~ 10/7)

현장에서 새 그리퍼로 데이터를 모으고 그 자리에서 학습한다. 예비 모델이 없으므로,
현장에서 막힐 만한 것을 대회 전에 전부 한 번씩 돌려 본다. 실제 물 따르기 성공률은 현장에서 처음 확인한다.

| # | 할 일 | 언제 | 문서 |
|---|---|---|---|
| 1 | 학습 파이프라인 리허설 (전송 → 학습 → 체크포인트 검증) | 화 | [`train/README.md`](train/README.md) |
| 2 | 1060 노트북 GPU 확인 | 접근 가능할 때 | 아래 |
| 3 | 새 그리퍼 + 손목 카메라 확인 (초점·해상도·컵 수위) | 월 | [`tools/CAMERA_CHECK.md`](tools/CAMERA_CHECK.md) |
| 4 | 녹화 명령 확인 (`lerobot-record`, 양팔 리더-팔로워) | 월 | 아래 |
| 5 | STT 실제 목소리 시험 | 화·수 | 아래 |

로봇 설정은 하나의 파일로 롤아웃·녹화가 같이 쓴다:
`cp pac2026/rollout/robot.env.example ~/UNITA_PAC2026/local/robot.env` 후 포트·카메라 경로를 채운다
(카메라 경로는 `python pac2026/tools/camera_check.py list`).

---

## 2. 1060 노트북 GPU 확인

**결론부터:** GTX 1060(Pascal, sm_61)은 **CUDA 12.6 빌드 torch(`+cu126`)** 만 쓸 수 있다.
CUDA 13 빌드(`+cu130`)와 CUDA 12.8·12.9 빌드는 Pascal 을 지원하지 않는다.
드라이버 535(CUDA 12.2)에서도 CUDA 12.x 빌드는 동작한다(같은 12.x 안의 호환).

```bash
cd ~/lerobot
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python pac2026/tools/offline_check.py --block-network
```

- `+cu126` 이고 `True GeForce GTX 1060` 이면 통과.
- `+cu130` / `+cu128` 이거나 `False` 이면 **인터넷 될 때** torch 를 바꾼다 (lerobot 0.6.1 요구 범위 torch ≥2.7,<2.12):

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

- 바꾼 뒤 다시 `offline_check.py` → "정책 로딩 + 더미 추론" 줄의 **청크 추론 시간**을 기록한다.
  - 정해진 양 따르기는 모델이 화면을 자주 봐야 한다(`--n-action-steps 10` ≈ 0.33 초마다).
    이때 추론 1회가 30 ms 안쪽이어야 30 fps 루프가 밀리지 않는다.
  - Whisper 를 GPU 에 같이 올릴지(`small` fp16 ≈ 1 GB)는 VRAM 6 GB 여유를 보고 정한다. 모자라면 STT 는 CPU.
- 현장은 오프라인 가정: 모델 캐시(`~/.cache/huggingface/hub/models--UNITAmanipulation--*`),
  Whisper(`~/.cache/whisper/small.pt`), resnet18 캐시(`~/.cache/torch/hub/checkpoints/`)를 미리 복사.
- `sudo apt install libportaudio2 alsa-utils` (마이크·스피커).

## 4. 녹화 명령 (`train/record.sh`)

`lerobot-record` 를 양팔(`bi_so_follower` + `bi_so_leader`)로 부르는 래퍼. 팔별 캘리브레이션(follower1/2, leader1/2)을
양팔 id 이름으로 자동 복사하고, 카메라는 학습 데이터와 같은 해상도(top 640×480, 손목 320×240, 30 fps)로 연다.

```bash
cd ~/lerobot
DRY_RUN=1 REPO_ID=UNITAmanipulation/pour_test bash pac2026/train/record.sh          # 명령만 확인
REPO_ID=UNITAmanipulation/pour_test NUM=2 EP_S=40 bash pac2026/train/record.sh       # 2개만 시험 녹화
```

- 녹화 중 키: → 다음 에피소드(일찍 끝내기), ← 방금 것 다시 녹화, Esc 중단.
- 리더암 1·2 가 왼팔·오른팔 어느 쪽인지 `robot.env` 에서 확인 필요 (기본: 1=왼팔).
- 물통 물 양 단계 등 에피소드 메모는 lerobot 이 저장하지 않으므로 표로 따로 적는다.
- 이 노트북에서 가짜 포트로 돌려 인자 해석·데이터셋 생성까지 통과하는 것을 확인함 (실제 녹화는 하드웨어 필요).

## 5. STT 실제 목소리 시험

녹음은 시연과 같은 마이크, 같은 녹음 코드로 한다. 사람당 약 10분.

```bash
cd ~/lerobot/pac2026
python stt/voice_command.py --list-devices                          # 마이크 번호 확인
python stt/record_clips.py --speaker 이름 --condition quiet         # 조용할 때 18문장
python stt/record_clips.py --speaker 이름 --condition noisy         # 시끄러울 때 (음악·말소리 틀어 놓고)
python stt/recognition_test.py ~/UNITA_PAC2026/local/outputs/stt/real_clips --device cpu
```

- 3명 이상, 마이크와 약 1 m, 현장처럼 주변 소리 있게.
- 합격 기준: **정지 누락 0**, **명령 아닌데 따르기 0**, 따르기 인식 95% 이상.
- 결과 CSV 와 로그를 Claude 에게 보내면 틀린 문장별로 키워드·프롬프트를 고친다.
- Enter 없는 자동 감지도 같이 시험: `bash run_demo.sh --replay` 실행 후 마이크에 "물 따라줘".
  시끄러운 데서 엉뚱하게 켜지면 `--vad-level 3`, 그래도 잦으면 `--stt-mode enter`.
