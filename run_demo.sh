#!/usr/bin/env bash
# PAC 2026 "말하면 따라주는 손" 데모 실행기.
#
#   bash pac2026/run_demo.sh                # 실제 로봇 (~/UNITA_PAC2026/local/robot.env 필요)
#   bash pac2026/run_demo.sh --sim          # 하드웨어 없이: 데이터셋 영상 + MuJoCo 관절 (근사)
#   bash pac2026/run_demo.sh --replay       # 하드웨어 없이: 데이터셋 재생 (open-loop)
#   bash pac2026/run_demo.sh --replay --episode 5 --no-stt     # 나머지 인자는 pour_rollout.py 로 전달
#
# 실행기 전용 옵션: --sim | --replay | --skip-check (오프라인 점검 생략) | --online (오프라인 환경변수 끔)
# 환경변수: UNITA_PY (파이썬 경로), UNITA_ROBOT_ENV (로봇 설정 파일, 기본 local/robot.env)
# 조작: 대기 중 "물 따라줘" 라고 말하기 (Enter 불필요, 키보드 p+Enter 도 됨; --stt-mode enter 면 Enter→말하기→Enter)
#       붓는 중 Space·s = 정지, q = 종료, Ctrl+C = 안전 종료
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # pac2026/
export UNITA_LOCAL="${UNITA_LOCAL:-$HOME/UNITA_PAC2026/local}"   # 결과물·robot.env 폴더 (깃 제외)
PY="${UNITA_PY:-$HOME/miniconda3/envs/lerobot/bin/python}"   # conda activate 대신 env 의 python 을 직접 사용
if [ ! -x "$PY" ]; then
  echo "[run_demo] 파이썬을 찾을 수 없음: $PY  (UNITA_PY=/경로/python 로 지정)" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

BACKEND=real
CHECK=1
ARGS=()
CHECK_ARGS=(--quick)
for a in "$@"; do
  case "$a" in
    --sim) BACKEND="replay+mujoco" ;;
    --replay) BACKEND=replay ;;
    --skip-check) CHECK=0 ;;
    --online) unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE ;;
    --no-stt) ARGS+=("$a"); CHECK_ARGS+=(--no-stt) ;;
    *) ARGS+=("$a") ;;
  esac
done
# --policy 를 주면 시작 전 점검도 같은 모델로 (체크포인트 폴더 등)
for i in "${!ARGS[@]}"; do
  case "${ARGS[$i]}" in
    --policy=*) CHECK_ARGS+=("${ARGS[$i]}") ;;
    --policy) CHECK_ARGS+=(--policy "${ARGS[$((i + 1))]:-}") ;;
  esac
done

EXTRA=()
if [ "$BACKEND" = real ]; then
  ENV_FILE="${UNITA_ROBOT_ENV:-$UNITA_LOCAL/robot.env}"
  if [ ! -f "$ENV_FILE" ]; then
    echo "[run_demo] 로봇 설정 없음: $ENV_FILE" >&2
    echo "           cp $HERE/rollout/robot.env.example $UNITA_LOCAL/robot.env 후 포트·카메라 경로를 채우세요." >&2
    echo "           하드웨어 없이 시험: --sim 또는 --replay" >&2
    exit 1
  fi
  set -a; . "$ENV_FILE"; set +a
  if grep -q REPLACE "$ENV_FILE"; then
    echo "[run_demo] $ENV_FILE 에 자리표시자(REPLACE)가 남아 있습니다." >&2
    exit 1
  fi
  # shellcheck disable=SC2206
  EXTRA=(${ROLLOUT_ARGS:-})
fi

if [ "$CHECK" = 1 ]; then
  echo "[run_demo] 오프라인 점검 (빠른 모드) ..."
  if ! "$PY" "$HERE/tools/offline_check.py" "${CHECK_ARGS[@]}"; then
    echo "[run_demo] 점검 FAIL. 원인을 고치거나 --skip-check 로 건너뛰세요." >&2
    exit 1
  fi
fi

echo "[run_demo] 롤아웃 시작 (backend=$BACKEND)"
exec "$PY" "$HERE/rollout/pour_rollout.py" --backend "$BACKEND" ${EXTRA[@]+"${EXTRA[@]}"} ${ARGS[@]+"${ARGS[@]}"}
