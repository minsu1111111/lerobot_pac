# stt — 음성 명령 인식

마이크로 받은 한국어 음성을 Whisper 로 받아쓰고, 명령어 사전으로 `pour`(따라줘) / `stop`(정지) /
`None`(명령 아님)을 돌려준다. PAC 2026 "말하면 따라주는 손" 데모의 앞단이며, 이전 색 분류 프로젝트의
[voice_command_vad.py](https://github.com/minsu1111111/lerobot_pac)를 명령어 인식용으로 고친 것이다.

```
Enter ─> 마이크(16kHz mono) ─> Enter ─> Whisper(ko, 어휘 프롬프트) ─> match_command ─> CommandResult
   └─ p / s / q + Enter (키보드 대체) ──────────────────────────────────────────────┘
```

**이 폴더는 혼자서 동작한다.** 다른 폴더(`paths.py`, `tts/`, `sim/` 등)를 import 하지 않으므로
`stt/` 폴더만 복사해서 다른 프로젝트에 넣어도 된다 (아래 "다른 프로젝트에 넣기").

| 파일 | 내용 |
|---|---|
| `voice_command.py` | 라이브러리(`VoiceCommander`, `match_command`) + CLI |
| `record_clips.py` | 실제 목소리 테스트 클립 녹음 (문장 안내 + Enter 녹음) |
| `recognition_test.py` | wav 폴더 인식률/지연시간 테스트 → CSV |
| `synth_test_clips.py` | edge-tts 합성 클립 생성 (사람 녹음 전 파이프라인 점검용, 선택) |
| `tests/` | 마이크·Whisper 없이 도는 단위 테스트 |
| `requirements.txt` | pip 의존성 |

## 설치

```bash
sudo apt install libportaudio2          # Ubuntu: sounddevice(마이크)가 필요로 함. Windows/macOS 는 불필요
pip install -r requirements.txt         # openai-whisper, numpy, scipy, sounddevice, webrtcvad
pip install "setuptools<81"             # webrtcvad 가 pkg_resources 를 못 찾을 때만
```

- torch 는 사용 중인 환경의 것을 그대로 쓴다 (openai-whisper 는 torch 버전을 고정하지 않음).
- Whisper `small`(≈460MB)은 첫 실행 때 `~/.cache/whisper` 로 받는다. **오프라인 현장에 가기 전에 한 번 실행**해 둔다.
- 마이크 없이 wav 파일만 인식(`--wav`, `transcribe_file`)할 때는 PortAudio 가 없어도 된다.

## 출력 폴더

`command.json`, 인식률 CSV, 녹음 클립의 기본 위치 (`voice_command.default_out_dir()`):

1. 환경변수 `UNITA_LOCAL` 이 있으면 `$UNITA_LOCAL/outputs/stt`
2. `stt/../../local` 폴더가 있으면 (또는 `~/UNITA_PAC2026/local`) `local/outputs/stt`
3. 그 밖에 (폴더만 복사해 간 경우) `stt/outputs` — `.gitignore` 에 들어 있다

## CLI

```bash
python voice_command.py                    # Enter 모드 반복 (q 또는 Ctrl+C 로 종료)
python voice_command.py --once             # 한 번만 (종료 코드 0 = 명령, 3 = 명령 없음, 1 = 오류)
python voice_command.py --stt-device cpu   # GPU 를 정책에 양보
python voice_command.py --mode vad         # 말소리 자동 감지 (조용할 때만)
python voice_command.py --wav a.wav b.wav  # 마이크 대신 파일 인식
python voice_command.py --list-devices     # 입력 장치 목록 → --mic 번호
python voice_command.py --help
```

Enter 모드에서는 녹음 대신 키보드로도 명령할 수 있다: `p`+Enter = 따라줘, `s`+Enter = 정지, `q`+Enter = 종료
(한글 자판 상태의 `ㅔ`/`ㄴ`/`ㅂ` 도 받음). 결과는 `<출력 폴더>/command.json` 에 원자적으로 저장된다.

## Python API

```python
from voice_command import VoiceCommander, match_command   # stt/ 를 sys.path 에 넣었을 때
# from stt import VoiceCommander                          # stt/ 의 부모 폴더가 sys.path 에 있을 때

vc = VoiceCommander(model="small", device="cpu")  # Whisper 로딩 + 워밍업은 여기서 한 번
r = vc.listen()          # Enter 로 녹음 → CommandResult(text, command, source, duration_s, stt_s)
if r.command == "pour": ...
elif r.command == "stop": ...
elif r.command is None:  # 인식 실패 → "다시 말씀해 주세요" 안내 후 다시 listen()
    ...
vc.poll_keyboard()       # 로봇 동작 중 non-blocking 으로 s+Enter(정지) 확인 → "stop" | None
r = vc.listen_auto()     # Enter 없이: 말소리가 들리면 자동 녹음 → 인식 (말이 없으면 계속 대기, p/s/q+Enter 도 받음)
vc.transcribe_file("a.wav")   # 파일 인식 (마이크 불필요)
match_command("물 좀 따라 줄래?")  # -> "pour"  (순수 함수, Whisper 불필요)
```

`VoiceCommander` 주요 인자: `model`(이름 또는 로딩한 모델 객체), `device`("auto"/"cpu"/"cuda"), `mic`(장치 번호/이름),
`max_record_s`, `initial_prompt`, `sample_len`(생성 토큰 상한, 기본 48), `num_threads`(변환할 때만 쓸 torch 스레드 수,
기본 torch 기본값; 전역 설정은 바꾸지 않고 변환이 끝나면 원래대로 돌려놓음), `warmup`.

명령어 사전은 `voice_command.py` 의 `COMMANDS` (+ "물"+요청 표현 조합 규칙, `COMMAND_EXCLUDE` 제외 표현).
팀 기본 명령은 **"따라줘"** / **"정지"** 이고 나머지는 동의어·오인식 대비다. 한 문장에 둘 다 나오면 stop 이 이긴다.

## 다른 프로젝트에 넣기

1. `stt/` 폴더를 통째로 복사한다 (예: `myproj/stt/`).
2. `pip install -r stt/requirements.txt` (+ Ubuntu 면 `sudo apt install libportaudio2`).
3. 코드에서:
   ```python
   import sys; sys.path.insert(0, "myproj/stt")
   from voice_command import VoiceCommander
   ```
   명령어를 바꾸려면 `match_command(text, commands=..., exclude=..., priority=...)` 에 자기 사전을 넘기거나
   `COMMANDS` / `INITIAL_PROMPT` 를 고친다.
4. 출력 폴더를 정하려면 `UNITA_LOCAL` 을 지정하거나 CLI 의 `--output` 을 쓴다.

## 테스트

```bash
cd stt && python -m pytest -q tests      # 마이크·Whisper·PortAudio 없이 돈다
python tests/test_voice_command.py       # pytest 가 없을 때
```

## 인식률 테스트

파일 이름 규칙 `<정답>__<화자>_<환경>_<번호>.wav` (정답 = `pour` / `stop` / `none`) 인 wav 폴더를 채점한다.

```bash
python recognition_test.py CLIPS --device cpu         # 정확도, 혼동표, stop 누락/오작동 pour, 지연(평균/중앙/p95/최대)
python recognition_test.py CLIPS --initial-prompt ""  # 프롬프트 효과 비교
python recognition_test.py CLIPS --model base         # 모델 크기 비교
python recognition_test.py CLIPS --threads 4          # CPU 스레드 수 비교
```

### 실제 목소리 녹음 프로토콜 (`record_clips.py`)

합성 음성(`synth_test_clips.py`)은 발음이 또렷해 실제보다 쉽다. 시연 전 판단은 실제 목소리로 한다.
`record_clips.py` 는 시연과 **같은 마이크·같은 Enter 녹음 코드**로 문장을 하나씩 안내하며 녹음한다.

```bash
python record_clips.py --list                                        # 문장 목록 (pour 6 / stop 6 / none 6)
python record_clips.py --speaker minsu --condition quiet --mic 3     # 18문장, 순서 섞음
python record_clips.py --speaker minsu --condition noisy --repeat 2  # 36문장
python recognition_test.py <출력 폴더>/real_clips --device cpu       # 채점 (현장 노트북이면 --device cuda 도)
```

한 문장마다 Enter = 녹음 시작 → 말하기 → Enter = 끝 → Enter = 저장 (`r` = 다시, `s` = 건너뜀, `q` = 종료).
너무 작거나 클리핑되면 경고가 나온다. 번호는 이어 붙이므로 덮어쓰지 않고, 문장은 `manifest.csv` 에 남는다.

- **화자 3명 이상** (남/여 섞어서, 가능하면 시연 담당자 포함) × 문장 18개 × **quiet / noisy** 2조건 → 108개 이상.
- **거리 약 1m**, 시연 때의 자세·목소리 크기, 시연에 쓸 마이크 그대로.
- **noisy**: 현장과 비슷한 소음 — 부스 웅성거림/음악을 스피커로 재생(대략 60~70dB)하거나 옆에서 대화.
- 합격 기준 (정확도보다 먼저 본다):
  - **stop 재현율 ≥ 99%** (stop 누락은 사실상 0건이어야 함)
  - **오작동 pour 0건** (stop/none 을 pour 로 인식하면 안 됨)
  - pour 인식 실패는 "다시 말씀해 주세요" 로 복구되므로 상대적으로 덜 위험하다.
- 기준에 못 미치면: 틀린 문장의 인식 텍스트(CSV)를 보고 `COMMANDS` 동의어/오인식 형태를 추가하거나
  `INITIAL_PROMPT` 를 조정한 뒤 **같은 클립으로 다시 채점**한다.

## 지연시간

Whisper 는 입력을 항상 30초 창으로 패딩해 인코더를 돌리므로, 짧은 명령어도 인코더 비용(CPU small ≈1.5초)은
고정이고 디코딩은 토큰당 60~120ms 다. 그래서 지연시간을 묶는 설정으로 부른다:

- `temperature=0.0` (재시도 없음, greedy) — Whisper 기본 재시도는 잡음에서 수십 초가 걸릴 수 있다.
- `without_timestamps=True` — 30초 창을 정확히 한 번만 디코딩 (타임스탬프가 있으면 남은 구간을 다시 디코딩하는 경우가 있음).
- `sample_len=48` — 환각으로 같은 말이 반복돼도 48토큰에서 멈춘다 (기본 224토큰이면 CPU 로 15~27초).

**CPU 경합 주의 (지연 튐의 실제 원인)**: 12~31초가 걸렸던 클립들을 다른 작업이 없을 때 다시 돌리면 모두
2.0~2.5초였고, 출력 토큰도 5~13개·세그먼트 1개로 정상이었다 (디코딩 반복/환각 아님). 같은 PC 에서 영상 렌더링 등
무거운 작업이 돌 때만 2초 → 10~20초로 튀며, 이때 Whisper 프로세스의 CPU 시간도 같은 비율로 늘어난다
(같은 출력에 CPU 를 5~6배 씀 = 코어를 나눠 쓰거나 전력/발열 한도로 코어가 느려짐). 스레드 수(10/6/4/2)를
바꿔도 튐은 남는다. 그래서:

- 시연 PC 에서는 STT 와 동시에 렌더링·학습 같은 무거운 작업을 돌리지 않는다. GPU 가 있으면 `device="cuda"`.
- CLI(`voice_command.py`, `recognition_test.py`)는 `OMP_WAIT_POLICY=PASSIVE` 를 기본으로 켠다 (이미 지정돼 있으면 그대로).
  바쁜 대기를 없애 명령 하나당 CPU 사용이 20 → 14 코어·초로 줄고 지연도 약간 준다. 라이브러리로 쓸 때는 호출 쪽
  프로세스 전체에 영향을 주므로 자동으로 켜지 않는다 — 원하면 torch import 전에 환경변수로 지정한다.

## 한계

- Whisper `small` CPU 변환에 명령 하나당 약 2~2.5초 (i5-1335U, 다른 작업 없을 때). GPU 면 훨씬 짧다.
- 한국어 전용 설정 (`language="ko"`). 명령어 사전 기반이라 사전에 없는 표현은 `None`.
- 합성 음성으로만 검증됨 — 실제 목소리·현장 소음 성능은 위 프로토콜로 확인해야 한다.
- 무음/작은 소리(최대 진폭 < 300)는 Whisper 를 건너뛰고 `None` (무음 환각 방지). 잡음에서는 Whisper 가
  "감사합니다" 같은 말을 지어낼 수 있으나 명령어가 아니면 `None` 이 된다.
- VAD(자동 감지)는 소음이 크면 잘못 켜지거나 녹음이 길어질 수 있다. 켜져도 명령어가 인식돼야 동작하므로 위험보다는
  "다시 말씀해 주세요" 가 잦아지는 쪽이다. 시끄러우면 `vad_level` 을 3 으로 올리거나 Enter 모드로 바꾼다.
  롤아웃(`rollout/pour_rollout.py`)은 기본이 자동 감지(`--stt-mode vad`)다.
- `sample_len=48` 이라 아주 긴 문장(대략 30음절 이상)은 뒤가 잘린다. 명령은 짧게 말하도록 안내한다.
