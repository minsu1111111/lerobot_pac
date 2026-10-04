# 새 그리퍼·손목 카메라 마운트 점검 (15분)

도구: `tools/camera_check.py`. 마운트를 확정하기 전에 네 가지를 본다.

1. 학습 때와 같은 해상도·fps 로 열리는지 (top 640×480, 손목 320×240, 30 fps, OpenCV)
2. 노출·화이트밸런스를 수동으로 고정할 수 있는지
3. 컵 거리(손목 카메라에서 5~15 cm)에서 초점이 맞는지
4. **핵심: 반투명 파란 컵 안의 물 높이가 손목 카메라에 보이는지** (목표 −1 cm / 목표 / +1 cm 를 구분할 수 있는지)

결과물은 `~/UNITA_PAC2026/local/outputs/camera_check/<날짜>/` 에 저장된다.

## 준비물

반투명 파란 컵, 물, 자, 네임펜이나 흰 마스킹테이프, 인쇄된 글씨 종이(초점용), 데모 때와 같은 조명.

## 절차

```bash
cd ~/UNITA_PAC2026/lerobot_pac
PY=~/miniconda3/envs/lerobot/bin/python
```

**0. 카메라를 쓰는 프로그램 전부 끄기 (lerobot-teleoperate·record, 브라우저 등).** 카메라 하나는 한 프로세스만 열 수 있다.

**1. 마운트 (2분).** 그리퍼와 손목 카메라 마운트를 달고 USB 를 꽂는다.

**2. 장치 확인 (1분)**

```bash
$PY tools/camera_check.py list
```

- 카메라마다 `video-index0`(영상)과 `video-index1`(메타데이터)이 보인다. **index0** 만 쓴다.
- 경로는 `/dev/v4l/by-id/...-video-index0` 로 적는다 (`/dev/videoN` 은 다시 꽂을 때마다 번호가 바뀜). 같은 모델 두 대의 by-id 가 겹치면 `/dev/v4l/by-path/` 를 쓴다.
- 어느 경로가 새 손목 카메라인지 모르겠으면 그 카메라만 뽑았다가 `list` 를 다시 돌려 사라지는 경로를 찾는다.
- 각 모드가 `OK` 로 나와야 한다. `불일치` 는 카메라가 그 해상도를 기본 지원하지 않는다는 뜻이고, 그러면 lerobot 이 에러를 낸다.
- `--formats` 를 붙이면 `v4l2-ctl --list-formats-ext`, `--ctrls` 를 붙이면 `--list-ctrls` 출력도 같이 나온다.

아래에서 `CAM=` 에 새 손목 카메라 경로를 넣는다.

```bash
CAM=/dev/v4l/by-id/usb-XXXX-video-index0
```

**3. 카메라별 check (2분)**

```bash
$PY tools/camera_check.py check --cam $CAM --width 320 --height 240           # 손목
$PY tools/camera_check.py check --cam $TOP --width 640 --height 480           # top
```

- `robot.env` 에 `CAM_FOURCC=MJPG` 를 썼다면 여기에도 `--fourcc MJPG` 를 붙인다.
- 카메라 세 대를 한 허브에 꽂았다면 터미널 세 개에서 **동시에** check 를 돌린다. 서로 다른 카메라는 동시에 열 수 있고, USB 대역폭이 모자란지는 세 대를 함께 돌려야 드러난다.

**4. 노출·화이트밸런스 고정 (3분).** 컵을 데모 위치에 두고 진행한다.

```bash
$PY tools/camera_check.py exposure --cam $CAM --sweep 80,150,250,333 --wb-temp 4600   # 후보 비교 (끝나면 원래대로 복구)
$PY tools/camera_check.py exposure --cam $CAM --exposure 250 --wb-temp 4600           # 고른 값 적용
```

- sweep 결과 PNG 에서 하얗게 날아간 곳이 없고 컵과 물 색이 가장 잘 보이는 값을 고른다.
- `exposure_time_absolute` 의 단위는 100 µs 이다. **30 fps 를 유지하려면 333 이하**여야 하고, 그보다 크면 fps 가 떨어진다 (지난 프로젝트 값 3000 은 300 ms 라서 약 3 fps 였다).
- 도구가 `exposure_dynamic_framerate=0` 도 같이 설정한다 (어두워도 fps 를 낮추지 않게).
- 적용한 명령은 `cam_setup_<카메라>.sh` 로 저장된다. **V4L2 설정은 USB 를 다시 꽂거나 재부팅하면 초기화되므로**, 매 세션 lerobot 을 실행하기 직전에 `bash .../cam_setup_<카메라>.sh` 를 실행한다.
- 자동으로 되돌리려면 `--reset` 을 쓴다.

**5. 초점 (2분).** 노출을 고정한 뒤에 한다 (선명도 숫자가 노출에 따라 달라짐).

```bash
$PY tools/camera_check.py focus --cam $CAM          # 창이 뜸. 창이 없으면 --no-window 로 콘솔에 0.5초마다 출력
```

- 인쇄 글씨 종이를 손목 카메라에서 약 10 cm 앞에 둔다. 렌즈를 돌려 숫자가 더 오르지 않는 지점에서 멈추고 `q` 로 끝낸다.
- 그다음 종이를 5 cm, 15 cm 로 옮겨서 글씨가 읽히는지 눈으로 본다.
- focus 를 마치면 최고값이 저장되고, 같은 날 `check` 는 그 값의 70% 이상이면 선명도 PASS 를 준다. 그러니 **종이를 그대로 둔 채** `check` 를 한 번 더 돌린다.

**6. 물 높이 3단계 촬영 (4분)**

- 컵에 목표 수위를 표시한다. 그리퍼(또는 시연 자세)로 컵을 **붓는 순간과 같은 카메라-컵 위치**에 두고, 촬영하는 동안 팔과 컵을 움직이지 않는다.
- 빈 컵부터 시작해서 물을 더 부어 가며 찍는다 (덜어내는 것보다 쉽다).

```bash
$PY tools/camera_check.py capture --cam $CAM --label empty
$PY tools/camera_check.py capture --cam $CAM --label level_minus1cm     # 목표 −1 cm 까지 채운 뒤
$PY tools/camera_check.py capture --cam $CAM --label level_target       # 목표까지
$PY tools/camera_check.py capture --cam $CAM --label level_plus1cm      # 목표 +1 cm 까지
```

라벨마다 3장씩 `<라벨>_NN.png` 로 저장된다. 같은 라벨로 다시 찍으면 번호가 이어지고 덮어쓰지 않는다.

**7. 비교 PNG 만들어 보내기**

```bash
$PY tools/camera_check.py compare --labels empty level_minus1cm level_target level_plus1cm
# 컵이 화면 중앙이 아니면 확대할 영역을 비율로 지정: --crop x,y,w,h  (예 --crop 0.3,0.4,0.4,0.4)
```

- 윗줄은 전체 화면(노란 상자는 확대 영역), 아랫줄은 확대 crop 이다.
- 확대는 nearest 보간이라 모델이 실제로 받는 320×240 픽셀 그대로 보인다.
- 출력된 `compare_....png` 를 Claude 에게 보낸다.

## 통과 기준

| 항목 | 기준 | 비고 |
|---|---|---|
| 해상도 | 정확히 일치 (손목 320×240, top 640×480) | 다르면 FAIL (lerobot 에러) |
| fps | 실측 ≥ 28 | 드랍 0 이 이상적 |
| 포화 | 하얗게 날아간 픽셀 < 2% | 컵 표면 반사 몇 점은 괜찮음 |
| 선명도 | focus 최고값의 ≥ 70%, 또는 focus 를 안 돌렸으면 ≥ 30 (320×240 기준) | 아래 설명 |
| 수동 노출/WB | exposure 실행 후 `auto_exposure=1(Manual)`, `white_balance_automatic=0` | 적용 후 줄에 표시됨 |
| **수위** | compare PNG 에서 −1 / 목표 / +1 cm 의 수면 위치나 색 경계가 눈으로 구분됨 | 사람이 판정 |

**선명도 숫자 읽는 법.** 선명도는 Laplacian 분산이고 **장면에 따라 달라진다.** 초점이 완벽해도 빈 벽을 보면 숫자가 낮다. 그래서 고정된 기준값보다, 같은 대상을 둔 채 focus 에서 찍은 최고값과 비교하는 쪽이 정확하다. 절대 하한 30 은 지난 프로젝트의 320×240 측정에서 정했다 (초점이 나간 손목캠 중앙값 9.5, 정상 씬캠 147). 같은 장면이라도 640×480 에서는 숫자가 2~4배 낮게 나온다 (노트북 웹캠: 320×240 에서 163, 640×480 에서 41). 그러니 top 카메라는 focus 최고값 기준으로만 판단한다.

## 실패하면

- **수위가 안 보이거나 세 장이 구분되지 않음**
  - 마운트를 아래로 기울여 카메라가 컵 안쪽이나 컵 옆면의 수면선을 보게 한다.
  - 컵 바깥쪽 카메라를 향한 면에 **흰 테이프로 목표선**을 붙인다. 수면이 선 위에 있는지 아래에 있는지가 기준이 된다.
  - 컵 뒤에 흰 종이 배경을 둔다.
  - 물 대신 **색 있는 음료**(주스, 식용색소 탄 물)를 쓴다. 파란 컵에는 주황·빨강 계열이 대비가 크다.
  - 반사광이 수면을 가리면 조명 방향을 바꾸거나 노출을 낮춘다.
- **fps < 28**: 노출 ≤ 333 인지, `exposure_dynamic_framerate=0` 인지 확인한다. 그다음 `--fourcc MJPG`, 카메라를 서로 다른 USB 포트(허브 말고 본체)에 꽂기 순서로 시도한다.
- **해상도 불일치**: 카메라가 320×240(또는 640×480)을 기본 지원하지 않는 것이다. `list --formats` 로 지원 목록을 확인하고 카메라를 바꾼다.
- **초점이 안 맞음**: 렌즈를 끝까지 돌려도 최고값이 낮거나 5 cm 와 15 cm 중 한쪽이 흐리면 고정초점 렌즈일 수 있다. 마운트를 컵에서 더 멀리 단다 (지난 프로젝트 손목캠은 5 cm 안에서만 초점이 맞아 접근 구간을 놓쳤다).
- **포화 ≥ 2%**: 노출을 낮춘다.
- **"열기 실패" / "프레임이 안 나옴"**: 다른 프로그램이 카메라를 쓰고 있지 않은지 본다 (사용 중인 프로세스가 있으면 도구가 출력한다). 또 index1(메타데이터) 경로를 쓰지 않았는지, USB 를 다시 꽂아 경로가 바뀌지 않았는지 확인한다.

## 주의

- 새 그리퍼와 손목 마운트로 바뀌면 손목 카메라 화면이 기존 학습 데이터와 달라진다. 그 팔은 데이터를 다시 모아야 할 가능성이 높다.
- 수집과 데모는 같은 노출/WB 값(`cam_setup_*.sh`)으로 맞춘다.
