#!/usr/bin/env bash
# 데스크톱 학습 결과의 체크포인트 하나(pretrained_model 만, ~200MB)를 노트북으로 가져오고 검증 명령을 출력.
#
#   DESKTOP=user@desktop-tailscale-name bash pac2026/train/fetch_checkpoint.sh <원격 run 폴더> [step|last|list]
#
#   <원격 run 폴더> : train.sh 의 OUT (예: '~/UNITA_PAC2026/local/train/act_pour_finetune_1004_1340')
#                     ~ 는 원격에서 풀리도록 따옴표로 감쌀 것
#   step            : 10000 처럼 숫자 (6자리 0 채움 → 010000), last (기본, 가장 최근 저장분), list (목록만)
#   DESKTOP=local   : 원격 대신 이 PC 의 경로에서 복사 (시험용)
#   DATASET=        : 출력할 검증 명령에 넣을 데이터셋 폴더 (기본: paths.DATASET)
#
# 결과: ${UNITA_LOCAL:-~/UNITA_PAC2026/local}/checkpoints/<run>/<step>/  ← 이 폴더 자체가 모델 폴더
#       (config.json, model.safetensors, policy_*processor*, train_config.json)
#       training_state/(옵티마이저, ~400MB)는 가져오지 않음 → 재개(RESUME)는 데스크톱에서만.
set -euo pipefail

RUN=${1:?사용법: DESKTOP=user@host $0 <원격 run 폴더> [step|last|list]}
STEP=${2:-last}
DESKTOP=${DESKTOP:?DESKTOP=user@desktop-tailscale-name 를 지정 (이 PC 경로면 DESKTOP=local)}
LOCAL_ROOT=${UNITA_LOCAL:-$HOME/UNITA_PAC2026/local}
DATASET=${DATASET:-$HOME/.cache/huggingface/lerobot/UNITAmanipulation/bi_so101_pour_water_20260920_194823}
RUN=${RUN%/}

remote() {  # 원격(또는 local) 에서 명령 실행. 경로의 ~ 가 원격 쪽에서 풀리도록 문자열로 넘김
    if [[ "$DESKTOP" == "local" ]]; then bash -c "$1"; else ssh "$DESKTOP" "$1"; fi
}

if [[ "$STEP" == "list" ]]; then
    remote "ls -l $RUN/checkpoints/"
    exit 0
elif [[ "$STEP" == "last" ]]; then
    STEP=$(remote "readlink $RUN/checkpoints/last") || { echo "[fetch] $RUN/checkpoints/last 없음 (아직 저장 전?)" >&2; exit 1; }
    STEP=$(basename "$STEP")
elif [[ "$STEP" =~ ^[0-9]+$ ]]; then
    STEP=$(printf '%06d' "$((10#$STEP))")   # lerobot: max(6, len(str(total_steps))) 자리
else
    echo "[fetch] step 은 숫자 / last / list" >&2; exit 1
fi

RUN_NAME=$(basename "$RUN")
DEST="$LOCAL_ROOT/checkpoints/$RUN_NAME/$STEP"
SRC="$RUN/checkpoints/$STEP/pretrained_model/"
[[ "$DESKTOP" == "local" ]] && SRC="${SRC/#\~/$HOME}" || SRC="$DESKTOP:$SRC"

mkdir -p "$DEST"
echo "[fetch] $SRC → $DEST"
rsync -a --info=progress2 "$SRC" "$DEST/"
[[ -f "$DEST/config.json" && -f "$DEST/model.safetensors" ]] || { echo "[fetch] 모델 파일이 없음: $DEST" >&2; exit 1; }
du -sh "$DEST"

PAC=$(cd "$(dirname "$0")/.." && pwd)
cat <<EOF

# 검증 (노트북, ~/lerobot 에서)
PY=~/miniconda3/envs/lerobot/bin/python
# 1) open-loop: 데이터셋 프레임 → 모델 → 완료 감지 (에피소드 몇 개만; 학습에 쓴 데이터라 파이프라인·감지 확인용)
\$PY $PAC/validate/offline_model.py --dataset $DATASET \\
    --model $DEST --episodes 0 10 20 30 40 --out $LOCAL_ROOT/outputs/offline_model_${RUN_NAME}_$STEP
# 2) 롤아웃 루프 (replay 백엔드, 하드웨어 없이)
\$PY $PAC/rollout/pour_rollout.py --backend replay --dataset $DATASET --policy $DEST \\
    --episode 0 --fast --auto-start --no-stt --no-tts --out $LOCAL_ROOT/outputs/rollout_${RUN_NAME}_$STEP
# 3) 실제 로봇 (나머지 인자는 pour_rollout.py 로 전달됨)
bash $PAC/run_demo.sh --policy $DEST
EOF
