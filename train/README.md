# train/ — 현장 데이터로 ACT 재학습 (노트북 → 데스크톱 → 노트북)

현장에서 그리퍼·손목 카메라 마운트가 바뀌면 `act_pour_water_100` 을 그대로 못 쓴다.
현장에서 새 데이터를 모으고, Tailscale 로 연결된 데스크톱(RTX 5060)에서 학습한 뒤, 중간 체크포인트를 노트북으로 가져와 검증·시연한다.

| 파일 | 내용 |
|---|---|
| `train.sh` | `lerobot-train` 래퍼. `MODE=finetune`(기본, `act_pour_water_100` 에서 이어 학습) / `MODE=scratch`. 명령을 먼저 출력하고, 출력 폴더가 이미 있으면 `RESUME=1` 이 아닌 한 거부 |
| `fetch_checkpoint.sh` | 데스크톱의 체크포인트 하나(`pretrained_model`, 약 200MB)를 `~/UNITA_PAC2026/local/checkpoints/<run>/<step>/` 로 rsync 하고 검증 명령 출력 |

명령은 LeRobot v0.6.1 소스(`configs/train.py`, `configs/default.py`, `scripts/lerobot_train.py`, `common/train_utils.py`)를 읽고 맞췄다.
이 노트북(CPU)에서 아주 짧게 실제로 돌려 확인했다 (맨 아래 "검증한 것").

아래 예시의 자리표시자:

```bash
DESKTOP=user@desktop-tailscale-name     # Tailscale MagicDNS 이름 또는 100.x.y.z (tailscale status 로 확인)
NEW=UNITAmanipulation/pour_venue_XXXX   # 현장에서 lerobot-record 할 때 쓴 --dataset.repo_id
PY=~/miniconda3/envs/lerobot/bin/python
```

---

## 0. 원래 모델은 어떻게 학습했나 (`act_pour_water_100` 의 train_config.json)

| 항목 | 값 |
|---|---|
| steps / batch_size | **100,000 / 16** (데이터셋 117,619 프레임 → 1 epoch ≈ 7,351 step, 전체 ≈ 13.6 epoch) |
| optimizer | AdamW, lr **1e-5** (backbone 도 1e-5), weight_decay 1e-4, grad_clip 10, scheduler 없음 |
| save_freq / log_freq | 10,000 / 50 |
| seed / num_workers | 1000 / 4 (`dataloader_multiprocessing_context=spawn`) |
| 정책 | ACT: chunk_size 100, n_action_steps 100, dim_model 512, enc 4 / dec 1, VAE(latent 32, kl_weight 10), dropout 0.1, ResNet18 (ImageNet 사전학습) |
| 기타 | image_transforms 끔, use_imagenet_stats=true, AMP 끔, device cuda, `push_to_hub=true` (repo_id=UNITAmanipulation/act_pour_water_100) |

`--policy.type=act` 의 기본값이 위 정책 값과 **전부 같다** (직접 비교함). 그래서 scratch 는 `--policy.type=act` + batch 16 + steps 만 주면 원래 학습과 같은 설정이 된다.
원래 모델은 우리 포크(v0.6.1)보다 새 LeRobot 으로 학습했다(config 에 `parallelism`, `ema`, `checkpoint_format` 같은 필드가 있음). 그래도 우리 포크에서 문제없이 불러와 학습했다.

---

## 1. 데스크톱 사전 준비 (리허설 전에 한 번)

```bash
ssh $DESKTOP                                   # Tailscale SSH 접속 확인
# 데스크톱에서:
cd ~/UNITA_PAC2026/lerobot_pac && git pull   # 이 저장소 (학습 PC 에도 clone). LeRobot 팀 포크 ~/lerobot 도 노트북과 같은 커밋으로
PY=~/miniconda3/envs/lerobot/bin/python

# (1) RTX 5060 = Blackwell sm_120. CUDA 12.8 이상으로 빌드된 torch 여야 한다
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
#   → cuda 12.8+ (노트북은 2.11.0+cu130), capability (12, 0), arch_list 에 'sm_120' 이 있어야 함
#   없으면: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  (cu130 도 됨)
#   torchcodec(영상 디코딩)도 torch 버전과 맞아야 함

# (2) 시작 모델 (finetune 용). 허브 캐시에 받아 두거나 ...
hf auth login                                  # 모델/데이터셋이 private 이면 필요
hf download UNITAmanipulation/act_pour_water_100
#   ... 또는 노트북에서 폴더째 복사 (-L: 허브 캐시는 심볼릭 링크라 실제 파일로 풀어야 함)
#   노트북: rsync -aL ~/.cache/huggingface/hub/models--UNITAmanipulation--act_pour_water_100/snapshots/*/ \
#               $DESKTOP:~/models/act_pour_water_100/
#   → 이 경우 BASE_MODEL=~/models/act_pour_water_100

# (3) scratch 를 할 거면 ResNet18 ImageNet 가중치 캐시 (현장에 인터넷 없으면 필수)
#   노트북: rsync -a ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth $DESKTOP:~/.cache/torch/hub/checkpoints/
#   finetune 은 train.sh 가 --policy.pretrained_backbone_weights=null 을 넣으므로 필요 없음 (확인함)

# (4) 디스크: 체크포인트 1개 = 592MB (모델 198MB + 옵티마이저 상태 394MB). 100k/5k 저장이면 20개 ≈ 12GB
df -h ~
```

학습은 SSH 가 끊겨도 계속 돌도록 **tmux 안에서** 실행한다: `tmux new -s train` (다시 붙기: `tmux a -t train`).

---

## 2. 현장 절차

### a. 데이터셋 전송 (노트북 → 데스크톱)

`lerobot-record` 는 기본으로 `~/.cache/huggingface/lerobot/<repo_id>/` 에 저장한다 (`--dataset.root` 를 줬으면 그 폴더).
이 폴더 안에 `meta/ data/ videos/` 가 바로 있다. 학습에서는 이 폴더를 `--dataset.root` 로 그대로 가리키면 된다 (위치는 어디든 됨).

**rsync (기본):**

```bash
# 노트북에서
SRC=~/.cache/huggingface/lerobot/$NEW
du -sh $SRC                                    # 기존 100 에피소드 = 1.6GB
rsync -a --info=progress2 --exclude .cache/ $SRC/ $DESKTOP:~/datasets/$(basename $NEW)/
# 데스크톱에서 DATASET_ROOT=~/datasets/<이름>, REPO_ID=$NEW
```

**HF Hub 경유 (인터넷이 될 때 대안):**

```bash
# 노트북: 업로드 (.cache/huggingface 는 hf 가 알아서 뺌)
hf upload $NEW ~/.cache/huggingface/lerobot/$NEW . --repo-type dataset --private
# 데스크톱: 원하는 폴더로 받기
hf download $NEW --repo-type dataset --local-dir ~/datasets/$(basename $NEW)
```

`hf upload` 로 올린 repo 엔 `v3.0` 태그가 없어서, `--dataset.root` 없이 lerobot 이 허브에서 직접 받으려 하면 실패한다
(`get_safe_version` → RevisionNotFoundError). 위처럼 `--local-dir` 로 받아 `--dataset.root` 로 가리키면 상관없다.
(`lerobot-record --dataset.push_to_hub=true` 로 올린 경우는 태그가 붙는다.)

> 새 데이터셋은 원래와 **같은 feature 이름**이어야 finetune 이 된다: `observation.images.top / left_wrist / right_wrist`, `observation.state`(12), `action`(12).
> finetune 은 입력 feature 목록을 데이터셋이 아니라 **시작 모델 config 에서** 가져온다(`policies/factory.py`). 카메라 해상도는 ACT 가 config 의 shape 를 쓰지 않아서 바뀌어도 동작은 하지만, 가능하면 같게 녹화할 것.

### b. 학습 (데스크톱)

```bash
cd ~/UNITA_PAC2026/lerobot_pac
export HF_HUB_OFFLINE=1                        # 현장 인터넷이 불안하면 (모델은 미리 캐시/복사해 둠)

# finetune (기본): act_pour_water_100 에서 시작, 정규화 통계는 새 데이터셋에서 다시 계산됨
REPO_ID=$NEW DATASET_ROOT=~/datasets/$(basename $NEW) JOB=venue_ft \
MODE=finetune STEPS=30000 SAVE_FREQ=5000 BATCH=16 NUM_WORKERS=8 \
bash train/train.sh

# scratch: 원래와 같은 하이퍼파라미터로 처음부터 (ResNet18 캐시 필요)
REPO_ID=$NEW DATASET_ROOT=~/datasets/$(basename $NEW) JOB=venue_scratch \
MODE=scratch STEPS=100000 SAVE_FREQ=10000 BATCH=16 NUM_WORKERS=8 \
bash train/train.sh
```

`STEPS=30000` 은 예시다 — 아래 c 의 속도 측정으로 정한다. 출력은 `~/UNITA_PAC2026/local/train/<JOB>/`, 로그는 그 옆 `<JOB>.log`.
`DRY_RUN=1` 을 앞에 붙이면 명령만 출력한다. 뒤에 붙인 인자는 lerobot-train 에 그대로 간다 (예: `'--dataset.episodes=[0,1,2,5]'` 로 나쁜 에피소드 빼기, 확인함).

train.sh 가 만드는 명령 (finetune; 이걸 직접 쳐도 된다):

```bash
~/miniconda3/envs/lerobot/bin/lerobot-train \
    --policy.path=UNITAmanipulation/act_pour_water_100 \
    --policy.pretrained_backbone_weights=null \
    --dataset.repo_id=$NEW \
    --dataset.root=$HOME/datasets/<이름> \
    --output_dir=$HOME/UNITA_PAC2026/local/train/venue_ft \
    --job_name=venue_ft \
    --steps=30000 --save_freq=5000 --batch_size=16 --num_workers=8 \
    --log_freq=50 --seed=1000 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false
```

scratch 는 첫 두 줄 대신 `--policy.type=act` 하나. 학습률을 바꾸려면 `LR=3e-5` (optimizer_lr 과 optimizer_lr_backbone 둘 다 설정). 기본은 원래와 같은 1e-5.

### c. 속도 재기 → STEPS 정하기

시작 후 1~2분(200 step 이상) 지나면 로그의 `step:` 줄 시각으로 속도를 잰다 (첫 줄은 워커 시작 시간이 섞여서 뺌):

```bash
LOG=~/UNITA_PAC2026/local/train/venue_ft.log
grep -aoE 'INFO [0-9-]+ [0-9:]+ \S+ step:[0-9]+' $LOG | tail -n 11 | awk '{split($3,t,":"); s=t[1]*3600+t[2]*60+t[3]; k=substr($5,6)+0; if(NR==2){s0=s;k0=k} if(NR>=2){s1=s;k1=k}} END{if(s1>s0) printf "%.3f steps/s (step %d→%d, %d s)\n",(k1-k0)/(s1-s0),k0,k1,s1-s0; else print "로그 줄이 더 필요함"}'
```

tqdm 진행 막대의 `it/s`(또는 `s/step`)를 봐도 된다. 그 다음:

```
STEPS ≈ steps/s × 학습에 쓸 수 있는 시간(초) × 0.9      (0.9 = 체크포인트 저장·검증 여유)
예) 5 steps/s, 3시간 → 5 × 10800 × 0.9 ≈ 48,000
epoch 수 = STEPS × BATCH / 데이터셋 프레임 수   (프레임 수는 로그의 dataset.num_frames)
```

이미 다른 STEPS 로 시작했다면 Ctrl+C 로 멈추고 같은 OUT 으로 `RESUME=1 STEPS=<새 값>` (아래 f). 마지막 저장 이후 step 은 잃는다.
로그 줄의 `data_s` 가 `updt_s` 보다 계속 크면 데이터 로딩(영상 디코딩)이 병목이다 → `NUM_WORKERS` 를 CPU 코어 수 가까이 올린다.
CUDA 면 로그에 `mem_gb`(GPU 메모리 최대치)도 나온다. OOM 이면 `BATCH=8` (RTX 5060 은 8GB 라 batch 16 이 들어가는지 리허설에서 꼭 볼 것).

### d. 모니터링과 중간 체크포인트 검증

```bash
# 데스크톱 상태 (노트북에서)
ssh $DESKTOP "tail -n 3 ~/UNITA_PAC2026/local/train/venue_ft.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
DESKTOP=$DESKTOP bash train/fetch_checkpoint.sh '~/UNITA_PAC2026/local/train/venue_ft' list

# 체크포인트 하나 가져오기 (step 번호 또는 last)
DESKTOP=$DESKTOP DATASET=~/.cache/huggingface/lerobot/$NEW \
bash train/fetch_checkpoint.sh '~/UNITA_PAC2026/local/train/venue_ft' 10000
#   → ~/UNITA_PAC2026/local/checkpoints/venue_ft/010000/   (이 폴더가 곧 모델 폴더)
```

fetch_checkpoint.sh 가 끝에 아래 검증 명령을 경로를 채워서 출력한다:

```bash
CK=~/UNITA_PAC2026/local/checkpoints/venue_ft/010000
DS=~/.cache/huggingface/lerobot/$NEW
# open-loop: 데이터셋 프레임 → 모델 → 완료 감지 (학습 데이터라 파이프라인·감지 확인용)
$PY validate/offline_model.py --dataset $DS --model $CK --episodes 0 10 20 30 40 \
    --out ~/UNITA_PAC2026/local/outputs/offline_model_venue_ft_010000
# 롤아웃 루프 (replay 백엔드)
$PY rollout/pour_rollout.py --backend replay --dataset $DS --policy $CK \
    --episode 0 --fast --auto-start --no-stt --no-tts
# 실제 로봇
bash run_demo.sh --policy $CK
```

`run_demo.sh` 는 `--policy` 를 pour_rollout.py 로 넘기지만, 시작 전 오프라인 점검(`tools/offline_check.py`)에는 넘기지 않아 점검은 기본 모델로 한다.
새 체크포인트를 점검하려면 따로: `$PY tools/offline_check.py --quick --block-network --policy $CK`.

`--model` / `--policy` 는 **`config.json` 이 들어있는 폴더**여야 한다. 데스크톱 경로를 직접 쓸 땐 `checkpoints/<step>/pretrained_model` 까지 붙일 것 (`checkpoints/<step>` 만 주면 실패).
같은 데이터 기준 원래 모델 결과와 비교: `$PY validate/offline_model.py --episodes 0 10 20 30 40` (기본 모델) 의 action MAE·감지율.

### e. 새 데이터셋으로 완료 감지 임계값

```bash
$PY analysis/joint_analysis.py --dataset ~/.cache/huggingface/lerobot/$NEW            # 먼저 결과만 보기
$PY analysis/joint_analysis.py --dataset ~/.cache/huggingface/lerobot/$NEW --install  # rollout/thresholds.json 갱신
```

`--install` 은 깃에 올라가는 `rollout/thresholds.json` 을 덮어쓴다. 시연하는 노트북에서 실행하고, 위 offline_model 검증은 임계값을 바꾼 뒤 다시 돌려 감지율을 본다.

### f. 끊긴 학습 이어가기 / STEPS 늘리기

```bash
OUT=~/UNITA_PAC2026/local/train/venue_ft RESUME=1 STEPS=50000 bash train/train.sh
# = lerobot-train --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json --resume=true --steps=50000
```

재개 때 설정은 체크포인트의 `train_config.json` 에서 읽고, CLI 인자만 덮어쓴다. 옵티마이저·RNG·데이터 순서까지 이어진다 (확인함: step 4 → 6).

---

## 3. 리허설 (기존 데이터셋으로 전체 흐름 미리 해 보기)

현장 전에 데스크톱에서 한 번: 전송 → 짧은 학습 → 속도 측정 → 체크포인트 가져오기 → 검증.

```bash
# 노트북: 기존 데이터셋 보내기
DS=UNITAmanipulation/bi_so101_pour_water_20260920_194823
rsync -a --info=progress2 --exclude .cache/ ~/.cache/huggingface/lerobot/$DS/ $DESKTOP:~/datasets/$(basename $DS)/

# 데스크톱 (tmux 안): 2000 step finetune, 1000 마다 저장
cd ~/UNITA_PAC2026/lerobot_pac
REPO_ID=$DS DATASET_ROOT=~/datasets/$(basename $DS) JOB=rehearsal_ft \
MODE=finetune STEPS=2000 SAVE_FREQ=1000 BATCH=16 NUM_WORKERS=8 bash train/train.sh
#   → 속도(c 의 명령), mem_gb, data_s vs updt_s 를 기록해 둔다 = 현장 STEPS 계산 근거
# scratch 속도도 한 번 (거의 같을 것): MODE=scratch STEPS=300 SAVE_FREQ=300 JOB=rehearsal_scratch ...
# 재개도 한 번: 위 학습 중 Ctrl+C → OUT=~/UNITA_PAC2026/local/train/rehearsal_ft RESUME=1 STEPS=2000 bash train/train.sh

# 노트북: 가져와서 검증
DESKTOP=$DESKTOP bash train/fetch_checkpoint.sh '~/UNITA_PAC2026/local/train/rehearsal_ft' last
CK=~/UNITA_PAC2026/local/checkpoints/rehearsal_ft/002000
$PY validate/offline_model.py --model $CK --episodes 0 10 20 30 40 --out ~/UNITA_PAC2026/local/outputs/offline_model_rehearsal
$PY rollout/pour_rollout.py --backend replay --policy $CK --episode 0 --fast --auto-start --no-stt --no-tts
```

기존 데이터셋 폴더에는 허브에서 받은 흔적(`.cache/huggingface/download/`)이 있다. 이 상태로 `--dataset.root` 없이 쓰면 lerobot 이 "옛 방식 다운로드" 로 보고 허브에서 다시 받으려 한다
(`has_legacy_hub_download_metadata`; 오프라인이면 실패, 확인함). train.sh 는 항상 `--dataset.root` 를 넣으므로 괜찮다. rsync 도 `--exclude .cache/` 로 뺀다.

---

## 4. 체크포인트 구조 (`common/train_utils.py`)

```
~/UNITA_PAC2026/local/train/<JOB>/            ← --output_dir (lerobot 이 만듦, 미리 있으면 에러)
└── checkpoints/
    ├── 005000/                                ← 6자리 0 채움 (steps 가 7자리 이상이면 그 자리수)
    │   ├── pretrained_model/                  ← 198MB. 검증·시연에 쓰는 폴더 (--model / --policy)
    │   │   ├── config.json  model.safetensors  train_config.json
    │   │   ├── policy_preprocessor.json   policy_preprocessor_step_3_normalizer_processor.safetensors   ← 새 데이터셋 통계
    │   │   └── policy_postprocessor.json  policy_postprocessor_step_0_unnormalizer_processor.safetensors
    │   └── training_state/                    ← 394MB. 재개용 (optimizer, rng, training_step.json)
    ├── 010000/ ...
    └── last -> 010000                         ← 상대 심볼릭 링크, 저장할 때마다 갱신
~/UNITA_PAC2026/local/train/<JOB>.log          ← train.sh 가 남기는 로그 (실행 명령 + 전체 출력)
```

마지막 step 은 save_freq 배수가 아니어도 항상 저장된다.

---

## 5. 주의할 점 (소스 확인)

- **push_to_hub:** 원래 모델 config 가 `push_to_hub=true`, `repo_id=UNITAmanipulation/act_pour_water_100` 이다. `--policy.path` 로 이어 학습하면서 `--policy.push_to_hub=false` 를 빼먹으면 **학습 끝에 원래 모델 repo 를 덮어쓴다.** scratch 도 기본이 `push_to_hub=true` 라 repo_id 가 없으면 시작부터 에러. train.sh 는 항상 false 로 넣는다. (저장된 config 에 원래 repo_id 가 남아 있지만 push 가 꺼져 있어 상관없음)
- **정규화 통계:** finetune(재개 아님)이면 정규화/역정규화 통계를 **새 데이터셋의 meta/stats.json** 으로 바꾼다 (`lerobot_train.py` 의 normalizer_processor override). 체크포인트의 통계가 데이터셋 통계와 같음을 확인함. 이미지는 ImageNet 평균/표준편차(`use_imagenet_stats=true`).
- **출력 폴더:** 이미 있으면 lerobot 이 `FileExistsError` (resume 아닐 때). 재개는 `--resume=true --config_path=<ckpt>/pretrained_model/train_config.json` 이고, `--output_dir` 이 아니라 config_path 로 찾는다. train.sh 는 모든 경로를 절대경로로 바꿔 넣는다 (재개 때 train_config.json 에 저장된 경로를 그대로 쓰므로).
- **--dataset.root:** `meta/info.json` 이 바로 들어있는 데이터셋 폴더 자체 (상위 폴더 아님). 없으면 `~/.cache/huggingface/lerobot/<repo_id>` 를 보고, 없거나 옛 다운로드 흔적이 있으면 허브에서 받으려 한다.
- **--policy.path 와 오프라인:** repo id 로 주면 config 를 `hf_hub_download` 로 읽는다. 인터넷이 없으면 `HF_HUB_OFFLINE=1` 을 켜거나 로컬 폴더 경로를 줄 것 (둘 다 확인함).
- **ResNet18 가중치:** scratch 는 torchvision ImageNet 가중치(`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`)가 필요하다 — 데스크톱에 없고 인터넷도 없으면 실패. finetune 은 train.sh 가 `pretrained_backbone_weights=null` 로 끈다 (체크포인트가 backbone 까지 덮어씀; 빈 TORCH_HOME 으로 학습해도 아무것도 안 받음을 확인). `BACKBONE_IMAGENET=1` 이면 끄지 않음.
- **wandb:** 기본 꺼짐이지만 명시적으로 `--wandb.enable=false`.
- **num_workers>0:** 워커를 spawn 으로 띄워서 첫 step 에 몇 초 걸린다 (`data_s` 6초). 속도 잴 때 첫 로그 줄은 뺄 것.
- **device:** 저장된 config.json 의 device 는 학습 때 값(cuda). offline_model.py·pour_rollout.py 는 `--device` 로 덮어써서 노트북 CPU 에서도 그대로 열린다 (확인함).

---

## 6. 이 노트북(CPU, i5-1335U)에서 검증한 것 — 2026-10-04

모두 기존 데이터셋, `HF_HUB_OFFLINE=1`, `DEVICE=cpu`. 로그: `~/UNITA_PAC2026/local/outputs/train_test/*.log` (체크포인트는 크기 때문에 지움).

| 시험 | 결과 |
|---|---|
| `OUT=…/train_test/ft_tiny MODE=finetune STEPS=4 SAVE_FREQ=2 BATCH=2 NUM_WORKERS=1 DEVICE=cpu LOG_FREQ=1 bash train/train.sh` | 시작 loss 0.034 (scratch 는 79) → 사전학습 가중치 로드됨. 4 step 뒤 가중치 최대 변화 4e-5. `checkpoints/000002`, `000004`, `last -> 000004` 생성, 각 592MB. 정규화 통계 = 데이터셋 stats.json |
| 같은 OUT 으로 다시 실행 | train.sh 가 거부 (RESUME=1 안내) |
| `OUT=…/ft_tiny RESUME=1 STEPS=6 bash train/train.sh --num_workers=0 --save_freq=6` | "Resuming data order at epoch 0, sample 8" → step 5, 6 → `000006`, `last -> 000006` |
| `offline_model.py --model …/000004/pretrained_model --episodes 0` | 감지 1/1, action MAE 1.44, 16초 |
| `pour_rollout.py --backend replay --policy …/000004/pretrained_model --episode 0 --fast --auto-start --no-stt --no-tts --test-stop-at 3` | 로드·워밍업 360ms, 정지=manual, 90 step |
| `DESKTOP=local fetch_checkpoint.sh '~/…/ft_tiny' [list\|last\|4]` | `checkpoints/ft_tiny/000006/` (198MB) 로 복사, 그 폴더로 offline_model·pour_rollout 둘 다 정상 |
| `MODE=scratch STEPS=2 SAVE_FREQ=2 BATCH=2 …` | 정상 (loss 79 → 66), 저장된 train_config 가 원래 것과 batch/steps/저장 주기/device/push 만 다름 |
| `BASE_MODEL=<로컬로 복사한 모델 폴더>` + 빈 `TORCH_HOME` | 같은 시작 loss 0.034, torch hub 다운로드 없음 |
| `'--dataset.episodes=[0,2,5]'` | 3 에피소드 4,873 프레임으로 학습 |

CPU 속도 (참고만, 대표값 아님): batch 2 에서 **≈0.4 steps/s** (step 당 2.4~2.9 s). 데스크톱 속도는 리허설에서 잴 것.

## 데스크톱에서 꼭 확인할 것

1. torch 가 CUDA 12.8+ 빌드이고 `sm_120` 지원 (`torch.cuda.get_arch_list()`), `torch.cuda.is_available()` 가 True. torchcodec 도 그 torch 와 맞는지 (리허설 학습이 돌면 OK)
2. 이 저장소(`~/UNITA_PAC2026/lerobot_pac`)와 팀 포크 `~/lerobot` 가 노트북과 같은 커밋, `lerobot-train` 이 `~/miniconda3/envs/lerobot/bin/` 에 있는지 (다르면 `LEROBOT_TRAIN=` 로 지정)
3. 데이터셋 경로: `DATASET_ROOT/meta/info.json` 이 있어야 함
4. 시작 모델이 캐시/로컬 폴더에 있는지 (현장 오프라인 대비), scratch 를 할 거면 ResNet18 캐시
5. batch 16 이 8GB 에 들어가는지 (`mem_gb`), steps/s, `data_s` vs `updt_s` → `NUM_WORKERS`
6. 디스크 여유 (체크포인트당 592MB)
