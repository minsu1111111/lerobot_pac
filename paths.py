"""경로 모음. 깃에 올리는 코드(이 폴더)와 올리지 않는 파일(local/)을 나눈다.

UNITA_PAC2026/
  lerobot_pac/  코드·설정·문서 (이 저장소, 깃에 올림)
  local/    결과물, 외부 모델 파일 (깃에 안 올림)

local/ 위치는 환경변수 UNITA_LOCAL 로 바꿀 수 있다 (저장소만 따로 clone 한 경우).
"""

import os
from pathlib import Path

GITHUB = Path(__file__).resolve().parent


def _local_root() -> Path:
    """결과물 폴더: UNITA_LOCAL → 저장소 옆 local/ → ~/UNITA_PAC2026/local → 이 폴더 안 local/ (gitignore)."""
    if os.environ.get("UNITA_LOCAL"):
        return Path(os.environ["UNITA_LOCAL"]).expanduser()
    for p in (GITHUB.parent / "local", Path.home() / "UNITA_PAC2026" / "local"):
        if p.is_dir():
            return p
    return GITHUB / "local"


LOCAL = _local_root()
os.environ.setdefault("UNITA_LOCAL", str(LOCAL))  # 같은 프로세스에서 쓰는 tts/stt/sim 도 같은 곳을 쓰게
OUTPUTS = LOCAL / "outputs"
THIRD_PARTY = LOCAL / "third_party"
SO101_DIR = THIRD_PARTY / "so101"  # TheRobotStudio/SO-ARM100 Simulation/SO101

HF_LEROBOT = Path.home() / ".cache/huggingface/lerobot"
DATASET = HF_LEROBOT / "UNITAmanipulation/bi_so101_pour_water_20260920_194823"
MODEL_REPO = "UNITAmanipulation/act_pour_water_100"
# 완료 감지 임계값 (깃에 올림). analysis/joint_analysis.py --install 로 갱신.
THRESHOLDS = GITHUB / "rollout" / "thresholds.json"
