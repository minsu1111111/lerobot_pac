# 캘리브레이션 (PAC 2026 대회 때 사용)

| 파일 | 장치 | 포트 | 캘리 날짜 |
|---|---|---|---|
| `follower1.json` | 왼팔 팔로워 (컵) | `/dev/follower1` | 2026-10-07 |
| `follower2.json` | 오른팔 팔로워 (물통) | `/dev/follower2` | 2026-10-09 |
| `leader1.json` | 왼팔 리더 | `/dev/leader1` | 2026-09-20 |
| `leader2.json` | 오른팔 리더 | `/dev/leader2` | 2026-09-18 |

쓰는 법:

```bash
C=~/.cache/huggingface/lerobot/calibration
mkdir -p $C/robots/so_follower $C/teleoperators/so_leader
cp follower1.json follower2.json $C/robots/so_follower/
cp leader1.json leader2.json $C/teleoperators/so_leader/
# 양팔 실행(bi_so_follower id=bimanual, bi_so_leader id=bimanual_leader)은 아래 이름을 읽는다
cp follower1.json $C/robots/so_follower/bimanual_left.json
cp follower2.json $C/robots/so_follower/bimanual_right.json
cp leader1.json $C/teleoperators/so_leader/bimanual_leader_left.json
cp leader2.json $C/teleoperators/so_leader/bimanual_leader_right.json
```

- 팔로워 두 대는 그리퍼 방향을 뒤집어 쓴다 (`gripper.drive_mode: 1`). `lerobot-calibrate` 를 다시 하면 0 으로 돌아가니 다시 1 로 바꿀 것.
- 모터 위치 기준(homing offset)이 팔마다 달라서 다른 로봇에는 그대로 쓸 수 없다. 같은 팔에 다시 꽂을 때만 쓴다.
