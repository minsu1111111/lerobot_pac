#!/usr/bin/env bash
# 양팔 리더-팔로워 데이터 수집 (lerobot-record, bi_so_follower + bi_so_leader).
# 포트·카메라·캘리브레이션 id 는 롤아웃과 같은 ~/UNITA_PAC2026/local/robot.env 에서 읽는다.
#
#   REPO_ID=UNITAmanipulation/pour_fixed_venue NUM=60 TASK="..." bash train/record.sh
#   DRY_RUN=1 bash train/record.sh          # 명령만 출력
#   RESUME=1 ... bash train/record.sh       # 같은 데이터셋에 이어서 녹화
#
# 설정 (환경변수): REPO_ID, NUM(에피소드 수, 기본 60), EP_S(에피소드 길이 초, 기본 60), RESET_S(기본 20),
#   TASK(작업 문장; ACT 는 안 쓰지만 데이터셋 메타에 남음), ROOT(저장 폴더, 기본 HF 캐시), PUSH(기본 false),
#   DISPLAY_DATA(기본 false), FOURCC.
# 물통 물 양 단계 같은 메모는 에피소드마다 local/outputs/record_notes.csv 에 직접 적는다 (lerobot 은 저장 안 함).
set -euo pipefail

PY_BIN="${UNITA_PY_BIN:-$HOME/miniconda3/envs/lerobot/bin}"
export UNITA_LOCAL="${UNITA_LOCAL:-$HOME/UNITA_PAC2026/local}"
ENV_FILE="${UNITA_ROBOT_ENV:-$UNITA_LOCAL/robot.env}"
[ -f "$ENV_FILE" ] || { echo "[record] 로봇 설정 없음: $ENV_FILE (rollout/robot.env.example 복사 후 채우기)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
if grep -q REPLACE "$ENV_FILE"; then echo "[record] $ENV_FILE 에 REPLACE 자리표시자가 남아 있음 (카메라 경로)" >&2; exit 1; fi

REPO_ID="${REPO_ID:?REPO_ID=팀/데이터셋이름 을 지정하세요}"
NUM="${NUM:-60}"; EP_S="${EP_S:-60}"; RESET_S="${RESET_S:-20}"
TASK="${TASK:-Pick up the cup with the left arm and pour water from the bottle into it with the right arm}"
PUSH="${PUSH:-false}"; FOURCC="${FOURCC:-${CAM_FOURCC:-}}"
CALIB="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}"

cam() {  # cam <경로> <폭> <높이>
  local f=""; [ -n "$FOURCC" ] && f=", \"fourcc\": \"$FOURCC\""
  printf '{"type": "opencv", "index_or_path": "%s", "width": %s, "height": %s, "fps": 30%s}' "$1" "$2" "$3" "$f"
}

# 팔 하나짜리 캘리브레이션(follower1.json 등)을 양팔 id 이름으로 복사 (없을 때만)
copy_calib() {  # copy_calib <robots|teleoperators> <so_follower|so_leader> <원본 id> <대상 id>
  local src="$CALIB/$1/$2/$3.json" dst="$CALIB/$1/$2/$4.json"
  [ -f "$dst" ] && return 0
  [ -f "$src" ] || { echo "[record] 캘리브레이션 없음: $src (rig_config/README.md 참고)" >&2; exit 1; }
  cp "$src" "$dst"; echo "[record] $3.json → $4.json"
}
if [ "${DRY_RUN:-0}" != 1 ]; then
  copy_calib robots so_follower "$CALIB_LEFT" "${ROBOT_ID}_left"
  copy_calib robots so_follower "$CALIB_RIGHT" "${ROBOT_ID}_right"
  copy_calib teleoperators so_leader "$LEADER_CALIB_LEFT" "${LEADER_ID}_left"
  copy_calib teleoperators so_leader "$LEADER_CALIB_RIGHT" "${LEADER_ID}_right"
fi

CMD=("$PY_BIN/lerobot-record"
  --robot.type=bi_so_follower --robot.id="$ROBOT_ID"
  --robot.left_arm_config.port="$LEFT_PORT" --robot.right_arm_config.port="$RIGHT_PORT"
  --robot.cameras="{\"top\": $(cam "$TOP_CAM" 640 480)}"
  --robot.left_arm_config.cameras="{\"wrist\": $(cam "$LEFT_CAM" 320 240)}"
  --robot.right_arm_config.cameras="{\"wrist\": $(cam "$RIGHT_CAM" 320 240)}"
  --teleop.type=bi_so_leader --teleop.id="$LEADER_ID"
  --teleop.left_arm_config.port="$LEADER_LEFT_PORT" --teleop.right_arm_config.port="$LEADER_RIGHT_PORT"
  --dataset.repo_id="$REPO_ID" --dataset.single_task="$TASK" --dataset.fps=30
  --dataset.num_episodes="$NUM" --dataset.episode_time_s="$EP_S" --dataset.reset_time_s="$RESET_S"
  --dataset.push_to_hub="$PUSH" --display_data="${DISPLAY_DATA:-false}")
[ -n "${ROOT:-}" ] && CMD+=(--dataset.root="$ROOT")
[ "${RESUME:-0}" = 1 ] && CMD+=(--resume=true)
CMD+=("$@")

printf '[record] 실행:\n  '; printf '%q ' "${CMD[@]}"; echo
[ "${DRY_RUN:-0}" = 1 ] && exit 0
echo "[record] 녹화 중 키: → 다음 에피소드(일찍 끝내기), ← 다시 녹화, Esc 중단 (lerobot-record 기본)"
exec "${CMD[@]}"
