# lerobot_pac — 음성 명령 인식 (UNITA Manipulation)

SO-101 로봇팔 2대로 물건을 분류/조작하는 시스템의 **음성 명령 앞단**입니다.
"빨간색 컵 집어줘"라고 말하면 텍스트로 바꾸고 색상 키워드를 뽑아 `command.json`에 저장합니다.
이후 단계(색상별로 학습된 LeRobot 정책 체크포인트 선택)는 이 파일을 읽어서 처리합니다.
이 스크립트는 로봇과 독립적으로 동작합니다.

- 지원 OS: Ubuntu 20.04 / 22.04, Windows 10 / 11
- Python 3.10+
- STT는 GPU가 있으면 GPU, 없으면 CPU로 동작

## 파이프라인

```
마이크 ──> 16kHz mono 변환 ──> webrtcvad 발화 감지 ──> Whisper(ko) ──> 색상 키워드 ──> command.json
```

| 단계 | 입력 | 처리 | 출력 |
|---|---|---|---|
| 1. 마이크 | 음성 | sounddevice로 장치가 지원하는 형식(int16) 그대로 읽음 | 장치 샘플레이트/채널의 PCM |
| 2. 형식 변환 | 1의 PCM | 스테레오→mono, 48kHz 등→16kHz (장치가 16kHz mono를 직접 주면 생략) | 30ms(480샘플) 프레임 |
| 3. VAD 녹음 | 프레임 | 최근 300ms 중 60% 이상이 음성이면 녹음 시작, 무음 800ms면 종료, 15초간 말이 없으면 타임아웃 | 발화 구간 PCM (시작 전 0.5초 포함, 끝 무음 0.3초만 유지) |
| 4. STT | PCM | Whisper `language="ko"` | 텍스트 |
| 5. 색상 추출 | 텍스트 | `COLOR_KEYWORDS`에서 가장 먼저 등장한 색 | `"red"` / `"blue"` / `"green"` / `"yellow"` / `None` |
| 6. 저장 | 텍스트 + 색상 | 임시파일에 쓴 뒤 교체 | `command.json` |

## 설치

### Ubuntu 20.04 / 22.04

```bash
sudo apt install libportaudio2 build-essential
pip install -r requirements.txt
```

- 20.04는 기본 Python이 3.8입니다. LeRobot conda 환경을 쓰거나 deadsnakes PPA로 python3.10(+ `-dev`, `-venv`)을 설치하세요.
- `webrtcvad` import 시 `No module named pkg_resources`가 나오면 `pip install "setuptools<81"`을 실행하세요.
- 실행 시 찍히는 ALSA/JACK 경고는 PortAudio 초기화 로그일 뿐이라 무시해도 됩니다.

### Windows 10 / 11

```powershell
pip install -r requirements.txt
```

- `requirements.txt`가 Windows에서는 자동으로 `webrtcvad-wheels`를 설치합니다(원본 `webrtcvad`는 C++ 빌드 도구가 필요).
- pip 기본 torch는 CPU 전용입니다. GPU를 쓰려면 [pytorch.org](https://pytorch.org)에서 CUDA 빌드를 설치하세요.
- **설정 > 개인 정보 및 보안 > 마이크 > "데스크톱 앱이 마이크에 액세스하도록 허용"** 이 꺼져 있으면 에러 없이 무음만 들어옵니다.

### 공통

- LeRobot 환경에 같이 설치해도 됩니다(openai-whisper는 torch 버전을 고정하지 않음).
- ffmpeg는 필요 없습니다(녹음 데이터를 numpy 배열로 Whisper에 직접 전달).
- Whisper 모델은 첫 실행 시 자동 다운로드됩니다(`small` 약 460MB).

## 사용법

```bash
python voice_command_vad.py --list-devices              # 마이크 목록 확인
python voice_command_vad.py --once --save-wav last.wav  # 한 번 인식 ("빨간색 컵 집어줘")
python voice_command_vad.py                             # 반복 인식, Ctrl+C로 종료
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | `small` | Whisper 모델 크기 (`tiny`/`base`/`small`/`medium`/`large`/`turbo`) |
| `--once` | off | 한 번만 인식하고 종료 |
| `--silence-ms` | `800` | 이 시간(ms) 동안 무음이면 녹음 종료 |
| `--vad-aggressiveness` | `2` | VAD 민감도 0(관대) ~ 3(엄격) |
| `--timeout` | `15` | 말소리가 없을 때 대기 시간(초) |
| `--frame-ms` | `30` | VAD 판정 프레임 길이 (10/20/30ms) |
| `--max-record-s` | `15` | 최대 녹음 길이(초). 소음 때문에 녹음이 안 끝나는 경우 대비 |
| `--mic` | 시스템 기본 | 입력 장치 번호 또는 이름 일부 |
| `--stt-device` | `auto` | Whisper 실행 장치 (`auto`/`cpu`/`cuda`) |
| `--output` | `command.json` | 결과 JSON 경로 |
| `--save-wav` | 없음 | 마지막 녹음을 wav로 저장 (튜닝/디버깅용) |

`--once` 종료 코드: `0` 색상 인식 / `3` 색상 키워드 없음 / `2` 타임아웃 / `1` 오류

## 출력 형식

```json
{
  "text": "빨간색 컵 집어줘",
  "color": "red",
  "timestamp": "2026-09-17T14:03:12+09:00"
}
```

- 색상을 못 찾으면 `"color": null`이 저장됩니다.
- 반복 모드에서는 인식할 때마다 덮어씁니다. 다음 단계에서는 `timestamp`가 바뀌면 새 명령으로 보고 `color`를 읽으면 됩니다.
- 색상 추가/삭제는 스크립트의 `COLOR_KEYWORDS` 딕셔너리만 수정하면 됩니다.

## 주의사항

- **현장이 시끄러우면 VAD가 오작동할 수 있습니다.** 잡음 때문에 녹음이 켜지거나 안 끝나면 `--vad-aggressiveness`를 높이고, 말하는 중간에 끊기면 `--silence-ms`를 늘려서 튜닝하세요.
- **안정성이 더 중요하면**(시연 등) 자동 감지 대신 Enter 키로 수동 시작/종료하는 방식을 쓰는 게 낫습니다.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 마이크를 열 수 없음 | `--list-devices`로 입력 장치 번호 확인 후 `--mic 번호` |
| 매번 타임아웃 + "입력이 완전히 0" 안내 | 음소거, Windows 마이크 권한, Ubuntu `pavucontrol` 입력 장치 |
| Ubuntu에서 `hw:` 장치가 안 열림 | `--mic pulse` 또는 `--mic default` |
| 인식 결과가 이상함 | `--save-wav last.wav`로 녹음이 잘리지 않았는지 먼저 확인 |
| CPU에서 너무 느림 | `--model base` (한국어 정확도는 낮아짐) |

## 테스트

마이크나 Whisper 없이 로직(색상 추출, JSON 저장, VAD 상태 머신, 리샘플링, 장치 설정 폴백)을 검증합니다.

```bash
python tests/test_voice_command_vad.py   # 또는 pytest tests/
```
