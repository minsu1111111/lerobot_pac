"""음성 명령 인식 (STT). stt/ 의 부모 폴더가 sys.path 에 있으면 `from stt import VoiceCommander`.

stt/ 폴더를 sys.path 에 직접 넣었다면 `from voice_command import VoiceCommander` (pour_rollout.py 방식).
"""

from .voice_command import (  # noqa: F401
    COMMANDS,
    INITIAL_PROMPT,
    CommandResult,
    VoiceCommander,
    default_out_dir,
    match_command,
)
