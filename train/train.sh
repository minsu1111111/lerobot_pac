#!/usr/bin/env bash
# ACT 학습 래퍼 (lerobot-train, LeRobot v0.6.1 소스 기준). 데스크톱(RTX 5060)에서 실행.
#
#   MODE=finetune  : --policy.path=$BASE_MODEL 에서 이어 학습 (정규화 통계는 새 데이터셋에서 다시 계산됨)
#   MODE=scratch   : --policy.type=act, 원래 act_pour_water_100 과 같은 하이퍼파라미터 (= ACT 기본값)
#   RESUME=1       : $OUT/checkpoints/last 에서 이어서 (같은 OUT, 끊긴 학습 재개 / STEPS 늘리기)
#
# 예)
#   DATASET_ROOT=~/datasets/pour_venue REPO_ID=UNITAmanipulation/pour_venue \
#   MODE=finetune STEPS=30000 SAVE_FREQ=5000 bash pac2026/train/train.sh
#   DRY_RUN=1 ... bash pac2026/train/train.sh          # 명령만 출력
#   bash pac2026/train/train.sh --log_freq=10          # 뒤에 붙인 인자는 lerobot-train 에 그대로 전달
#
# 환경변수 (기본값):
#   MODE=finetune | scratch
#   REPO_ID=UNITAmanipulation/bi_so101_pour_water_20260920_194823   (데이터셋 이름; 로컬 폴더면 허브 접속 안 함)
#   DATASET_ROOT=~/.cache/huggingface/lerobot/$REPO_ID              (meta/ data/ videos/ 가 바로 들어있는 폴더)
#   BASE_MODEL=UNITAmanipulation/act_pour_water_100                 (finetune 시작점: HF repo id 또는 pretrained_model 폴더)
#   JOB=act_pour_${MODE}_<날짜시각>
#   OUT=${UNITA_LOCAL:-~/UNITA_PAC2026/local}/train/$JOB            (없어야 함. 있으면 RESUME=1 이 아닌 한 거부)
#   STEPS=100000  SAVE_FREQ=5000  BATCH=16  NUM_WORKERS=4  DEVICE=cuda  LOG_FREQ=50  SEED=1000
#   LR=            (비우면 ACT 기본 1e-5 = 원래 학습값. 넣으면 optimizer_lr 과 optimizer_lr_backbone 둘 다)
#   BACKBONE_IMAGENET=0   finetune 에서 ResNet18 ImageNet 가중치 로드 생략(어차피 체크포인트가 덮어씀 → 오프라인 안전)
#                         scratch 는 항상 ImageNet 가중치 사용 (~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth 필요)
#   LEROBOT_TRAIN=~/miniconda3/envs/lerobot/bin/lerobot-train (없으면 PATH 의 lerobot-train)
#   DRY_RUN=1      명령만 출력하고 종료
set -euo pipefail

MODE=${MODE:-finetune}
REPO_ID=${REPO_ID:-UNITAmanipulation/bi_so101_pour_water_20260920_194823}
DATASET_ROOT=${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/$REPO_ID}
BASE_MODEL=${BASE_MODEL:-UNITAmanipulation/act_pour_water_100}
BASE_MODEL=${BASE_MODEL/#\~/$HOME}
JOB=${JOB:-act_pour_${MODE}_$(date +%m%d_%H%M)}
LOCAL_ROOT=${UNITA_LOCAL:-$HOME/UNITA_PAC2026/local}
OUT=${OUT:-$LOCAL_ROOT/train/$JOB}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-5000}
BATCH=${BATCH:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda}
LOG_FREQ=${LOG_FREQ:-50}
SEED=${SEED:-1000}
LR=${LR:-}
BACKBONE_IMAGENET=${BACKBONE_IMAGENET:-0}
RESUME=${RESUME:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "${LEROBOT_TRAIN:-}" ]]; then
    if [[ -x "$HOME/miniconda3/envs/lerobot/bin/lerobot-train" ]]; then
        LEROBOT_TRAIN="$HOME/miniconda3/envs/lerobot/bin/lerobot-train"
    else
        LEROBOT_TRAIN=$(command -v lerobot-train || true)
    fi
fi
[[ -n "$LEROBOT_TRAIN" ]] || { echo "[train] lerobot-train 을 찾을 수 없음 (LEROBOT_TRAIN=... 로 지정)" >&2; exit 1; }

# 절대경로로 (resume 때 checkpoint 의 train_config.json 에 저장된 경로를 그대로 다시 쓰므로)
DATASET_ROOT=$(realpath -m "${DATASET_ROOT/#\~/$HOME}")
OUT=$(realpath -m "${OUT/#\~/$HOME}")
LOG="${OUT}.log"   # OUT 은 lerobot 이 직접 만들어야 하므로(이미 있으면 에러) 로그는 옆에 둔다

CMD=("$LEROBOT_TRAIN")
if [[ "$RESUME" == "1" ]]; then
    CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
    [[ -f "$CFG" ]] || { echo "[train] RESUME=1 인데 $CFG 가 없음" >&2; exit 1; }
    # 재개 시 설정은 체크포인트의 train_config.json 에서 읽고 CLI 인자가 덮어씀 (configs/train.py)
    CMD+=("--config_path=$CFG" "--resume=true" "--steps=$STEPS")
else
    if [[ -e "$OUT" ]]; then
        echo "[train] 출력 폴더가 이미 있음: $OUT" >&2
        echo "        끊긴 학습을 이어가려면 RESUME=1, 새로 하려면 JOB/OUT 을 바꿀 것 (lerobot 도 FileExistsError 로 거부함)" >&2
        exit 1
    fi
    [[ -f "$DATASET_ROOT/meta/info.json" ]] || {
        echo "[train] 데이터셋 폴더가 아님 (meta/info.json 없음): $DATASET_ROOT" >&2
        echo "        DATASET_ROOT 는 meta/ data/ videos/ 가 바로 들어있는 폴더여야 함" >&2
        exit 1
    }
    case "$MODE" in
        finetune)
            CMD+=("--policy.path=$BASE_MODEL")
            # 원래 모델 config 는 push_to_hub=true, repo_id=UNITAmanipulation/act_pour_water_100 이다.
            # 아래 push_to_hub=false 가 없으면 학습 끝에 원래 모델 repo 를 덮어쓴다!
            [[ "$BACKBONE_IMAGENET" == "1" ]] || CMD+=("--policy.pretrained_backbone_weights=null")
            ;;
        scratch)
            # ACT 기본값 = 원래 학습값 (chunk 100, n_action_steps 100, dim 512, enc 4 / dec 1, VAE, kl 10, lr 1e-5, ResNet18 ImageNet)
            CMD+=("--policy.type=act")
            ;;
        *) echo "[train] MODE 는 finetune 또는 scratch" >&2; exit 1 ;;
    esac
    CMD+=(
        "--dataset.repo_id=$REPO_ID"
        "--dataset.root=$DATASET_ROOT"
        "--output_dir=$OUT"
        "--job_name=$JOB"
        "--steps=$STEPS"
        "--save_freq=$SAVE_FREQ"
        "--batch_size=$BATCH"
        "--num_workers=$NUM_WORKERS"
        "--log_freq=$LOG_FREQ"
        "--seed=$SEED"
        "--policy.device=$DEVICE"
        "--policy.push_to_hub=false"
        "--wandb.enable=false"
    )
    if [[ -n "$LR" ]]; then
        CMD+=("--policy.optimizer_lr=$LR" "--policy.optimizer_lr_backbone=$LR")
    fi
fi
CMD+=("$@")

echo "[train] MODE=$MODE RESUME=$RESUME"
echo "[train] dataset: $REPO_ID @ $DATASET_ROOT"
echo "[train] output : $OUT   (로그: $LOG)"
echo "[train] 명령:"
printf '  %q' "${CMD[0]}"; printf ' \\\n    %q' "${CMD[@]:1}"; echo
[[ "$DRY_RUN" == "1" ]] && exit 0

mkdir -p "$(dirname "$OUT")"
{ echo "# $(date '+%F %T') $(hostname)"; printf '%q ' "${CMD[@]}"; echo; } >> "$LOG"
# PYTHONUNBUFFERED: tee 로 넘겨도 로그가 바로바로 찍히게
PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
