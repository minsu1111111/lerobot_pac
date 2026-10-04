"""tts/ 폴더만 있어도 테스트가 돌도록 상위 폴더(tts/)를 import 경로에 넣는다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
