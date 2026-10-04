# tts — 고정 문구 음성 안내 (오프라인 재생)

로봇 제어 루프(30 fps)를 막지 않고 짧은 안내 음성을 틀어 주는 모듈입니다.
문장은 미리 wav 로 만들어 두고(인터넷 필요, 1회), 실행 중에는 그 wav 만 재생합니다(오프라인).

**독립 폴더**: 이 폴더만 다른 프로젝트에 복사해도 동작합니다. 재생 쪽은 파이썬 표준 라이브러리만 쓰고,
이 저장소의 `paths.py`, lerobot, 다른 폴더를 import 하지 않습니다.

| 파일 | 내용 |
|---|---|
| `tts_player.py` | `TTSPlayer` — 백그라운드 스레드 + 큐로 wav 재생, 실패 시 자막만 |
| `phrases.py` | 기본 문구 `PHRASES`(키 → 문장), `CRITICAL_KEYS`, 문구 파일 읽기 `load_phrases` |
| `generate_wavs.py` | edge-tts 로 문구 → `wavs/<key>.wav` 생성 (24 kHz 모노 16-bit, 무음 제거·음량 정규화) |
| `wavs/` | 생성된 wav (깃에 커밋, 약 700 KB) |
| `tests/` | 오디오 장치 없이 도는 pytest |

이 프로젝트의 기본 문구: `ready` `start` `done` `stopped` `retry` `timeout`
(키는 `rollout/pour_rollout.py`, `tools/mux_tts_audio.py` 가 그대로 씁니다. 바꾸지 마세요.)
30 fps 루프와 묶은 타이밍 시험은 이 폴더 밖 `tools/tts_loop_timing.py` 에 있습니다.

## 설치

재생만 할 때 — 파이썬 패키지 불필요. 시스템 재생 프로그램 하나 이상:

```bash
sudo apt install alsa-utils        # aplay
sudo apt install pulseaudio-utils  # paplay (Ubuntu 데스크톱에는 보통 둘 다 있음)
```

wav 를 새로 만들 때만 (인터넷 필요):

```bash
pip install -r requirements.txt    # edge-tts, av, numpy
```

## wav 만들기

```bash
python generate_wavs.py                                  # phrases.py 전체 → wavs/
python generate_wavs.py --keys done start                # 일부만
python generate_wavs.py --voice ko-KR-InJoonNeural --rate +5%
python generate_wavs.py --phrases my_phrases.json --out my_wavs   # 다른 문장 세트
```

`--phrases` 를 주면 `--out` 이 필수이고(기본 `wavs/` 덮어쓰기 방지), 출력 폴더에 `phrases.json`
(문구 목록)도 함께 저장됩니다. 음성 목록은 `edge-tts --list-voices` (영어 등 다른 언어도 가능).

문구 파일 형식:

```json
{"phrases": {"hello": "안녕하세요.", "bye": "안녕히 가세요."}, "critical": ["bye"]}
```

또는 `{키: 문장}` 만 있는 JSON, 또는 `PHRASES = {...}` (선택 `CRITICAL_KEYS = {...}`) 가 있는 `.py`.
키는 파일 이름(`<key>.wav`)이 됩니다.

## 재생 확인

```bash
python tts_player.py                 # 전체 문구 재생
python tts_player.py start done      # 일부
python tts_player.py --no-audio      # 소리 없이 자막만
python tts_player.py --device plughw:1,0
python tts_player.py --wav-dir my_wavs
```

## 파이썬 API

```python
from tts_player import TTSPlayer

tts = TTSPlayer()                     # 기본: 이 폴더의 wavs/ + phrases.py
text = tts.say("start")               # 즉시 반환 (자막 문자열). 재생은 백그라운드
...                                   # 제어 루프 계속
tts.say("done")
tts.wait(timeout=6.0)                 # 대기열·재생이 끝날 때까지 (끝나면 True)
tts.close()                           # 또는 with TTSPlayer() as tts: ...
```

생성자 인자 (모두 선택, 기본값은 예전과 동일):

| 인자 | 기본 | 설명 |
|---|---|---|
| `wav_dir` | `<이 폴더>/wavs` | `<key>.wav` 가 있는 폴더 |
| `phrases` | 아래 참고 | 자막 문구 `{key: text}` 또는 문구 파일 경로(.json/.py) |
| `critical_keys` | 문구 파일의 `critical`, 없으면 `CRITICAL_KEYS` | 대기열에서 버리지 않을 키 |
| `device` | `None`(시스템 기본) | `aplay -D` / `paplay --device` 값 |
| `enabled` | `True` | `False` 면 소리 없이 자막만 |
| `max_pending` | `3` | 대기열 길이 상한 |
| `backends` | `("aplay", "paplay")` | 시도 순서. 명령 이름 또는 함수 `fn(pcm: bytes, sr: int, device)` |

`phrases=None` 이면 `wav_dir/phrases.json` 이 있으면 그것을, 없으면 `phrases.py` 의 `PHRASES` 를 씁니다.
그래서 `generate_wavs.py --phrases ... --out DIR` 로 만든 폴더는 `TTSPlayer(wav_dir=DIR)` 만으로 자막까지 맞습니다.

유용한 속성: `tts.played` (실제 재생된 `(key, backend)` 기록), `tts.backend` (현재 백엔드, 음소거면 `None`),
`tts.clips[key].seconds` (문구 길이), `tts.phrases`.

### 동작 방식

- **비차단**: `say()` 는 텍스트를 출력·반환하고 큐에 넣기만 합니다 (측정 약 0.02 ms). wav 는 생성 시 메모리에 올려 두고 재생 프로그램의 표준입력으로 raw PCM 을 넣습니다.
- **큐**: 재생 중인 문구를 끊지 않고 줄 세웁니다. 같은 키가 이미 대기 중이면 무시합니다(중복 안내 방지).
- **`CRITICAL_KEYS`**: 대기열이 `max_pending` 을 넘으면 가장 오래된 일반 문구부터 버리고, `done`/`stopped`/`timeout` 같은 중요 문구는 버리지 않습니다.
- **실패 처리**: `aplay` 가 실패하면 `paplay` 로 같은 문구를 다시 시도하고, 모두 실패하면 경고를 한 번만 출력하고 이후 자막만 표시합니다. 제어 루프로 예외를 올리지 않습니다.
- **wav 누락**: 생성 시점에 `FileNotFoundError` (어떤 키가 어느 폴더에 없는지 표시).

### 오디오 장치 고르기

```bash
aplay -l                     # 카드/장치 번호 확인 → device="plughw:1,0"
pactl list short sinks       # PulseAudio 싱크 이름 → paplay 쪽 device
```

`device` 는 첫 번째로 쓰는 백엔드 형식에 맞춰야 합니다. PulseAudio 싱크 이름을 쓰려면
`TTSPlayer(device="<sink>", backends=("paplay",))` 처럼 백엔드를 지정하세요.

## 다른 프로젝트에 넣기

1. 이 폴더를 통째로 복사합니다 (예: `myproj/tts/`).
2. 다른 문장을 쓰려면 `phrases.py` 를 고치지 말고 문구 파일을 만들어 wav 를 생성합니다:
   `python tts/generate_wavs.py --phrases myproj/voice.json --out myproj/voice_wavs`
3. 코드에서:

```python
import sys
sys.path.insert(0, "path/to/tts")          # 또는 패키지로: from tts.tts_player import TTSPlayer
from tts_player import TTSPlayer

tts = TTSPlayer(wav_dir="myproj/voice_wavs")   # phrases.json 을 자동으로 읽음
tts.say("hello")
```

다른 오디오 라이브러리를 쓰고 싶으면 함수 백엔드를 넘깁니다:

```python
def play(pcm: bytes, sr: int, device):     # 끝날 때까지 블록, 실패하면 예외
    ...
tts = TTSPlayer(backends=(play, "aplay"))
```

## 테스트

```bash
cd tts && python -m pytest -q tests        # 오디오 장치·인터넷 불필요
```

## 한계

- 고정 문구만 재생합니다 (실시간 합성 아님). 문구를 바꾸면 인터넷 되는 곳에서 wav 를 다시 만들어야 합니다.
- wav 는 16-bit 모노만 받습니다 (`generate_wavs.py` 출력 형식).
- 재생 중인 문구를 중간에 끊는 API 는 없습니다 (`close()` 만 끊음). 우선순위 끼어들기 없음.
- 기본 백엔드는 Linux(`aplay`/`paplay`) 전용입니다. 다른 OS 는 함수 백엔드를 넘기거나 자막만 동작합니다.
- `pw-play`(PipeWire)는 Ubuntu 22.04 버전이 표준입력 재생을 못 해서 기본 목록에서 뺐습니다.
- edge-tts 는 Microsoft 온라인 서비스를 씁니다 (생성할 때만). 생성된 음성의 사용 조건은 직접 확인하세요.
