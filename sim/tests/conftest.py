"""sim/ 폴더 단독으로 pytest 를 돌릴 수 있게 sim/ 을 import 경로에 넣는다.
렌더링은 쓰지 않는다 (render="none"). SO-101 자산이 없으면 모델이 필요한 시험은 건너뛴다."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def so101():
    from sim_paths import so101_dir

    try:
        return so101_dir()
    except FileNotFoundError as e:
        pytest.skip(f"SO-101 자산 없음 (python fetch_so101.py 또는 SO101_DIR): {e}")


@pytest.fixture(scope="session")
def model(so101):
    from bi_so101_scene import build_model

    return build_model(props=False, so101=so101)
