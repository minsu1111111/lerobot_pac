"""sim/ 폴더 안에서만 쓰는 경로 규칙. 이 폴더만 다른 프로젝트로 복사해도 동작하도록 바깥 모듈을 import 하지 않는다.

SO-101 공식 MJCF/STL 폴더 (so101_new_calib.xml, assets/):
  1. 환경변수 SO101_DIR
  2. $UNITA_LOCAL/third_party/so101      (UNITA_LOCAL 환경변수가 있을 때; fetch_so101.py 기본 위치와 맞춤)
  3. <sim>/third_party/so101            (fetch_so101.py 를 이 폴더 단독으로 돌린 경우)
  4. <sim>/../../local/third_party/so101 또는 ~/UNITA_PAC2026/local/third_party/so101 (프로젝트 local/ 이 있을 때)
  5. 없으면 fetch_so101.py 를 돌리라는 오류

결과물 폴더:
  1. 환경변수 SIM_OUTPUTS
  2. $UNITA_LOCAL/outputs/sim
  3. <sim>/../../local 또는 ~/UNITA_PAC2026/local 아래 outputs/sim (있을 때)
  4. <sim>/outputs
"""

import os
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
# 프로젝트 local/ 후보 (있을 때만 씀): 저장소 옆 local/, ~/UNITA_PAC2026/local
_LOCAL_CANDS = [SIM_DIR.parents[1] / "local", Path.home() / "UNITA_PAC2026" / "local"]
_PROJECT_LOCAL = next((p for p in _LOCAL_CANDS if p.is_dir()), _LOCAL_CANDS[0])
ARM_XML_NAME = "so101_new_calib.xml"


def local_root() -> Path | None:
    """UNITA_PAC2026 의 local/ 폴더 (UNITA_LOCAL 환경변수 우선). 없으면 None."""
    if os.environ.get("UNITA_LOCAL"):
        return Path(os.environ["UNITA_LOCAL"]).expanduser()
    return _PROJECT_LOCAL if _PROJECT_LOCAL.is_dir() else None


def so101_dir() -> Path:
    """SO-101 자산 폴더를 찾는다. 없으면 FileNotFoundError (한국어 안내)."""
    env = os.environ.get("SO101_DIR")
    if env:
        p = Path(env).expanduser()
        if not (p / ARM_XML_NAME).is_file():
            raise FileNotFoundError(f"SO101_DIR={p} 에 {ARM_XML_NAME} 가 없습니다. 경로를 확인하거나 "
                                    f"'python {SIM_DIR / 'fetch_so101.py'}' 로 받으세요.")
        return p
    cands = [SIM_DIR / "third_party" / "so101", _PROJECT_LOCAL / "third_party" / "so101"]
    if os.environ.get("UNITA_LOCAL"):
        cands.insert(0, Path(os.environ["UNITA_LOCAL"]).expanduser() / "third_party" / "so101")
    for p in cands:
        if (p / ARM_XML_NAME).is_file():
            return p
    raise FileNotFoundError(
        "SO-101 MuJoCo 모델 파일(so101_new_calib.xml)을 찾지 못했습니다.\n"
        f"  찾아본 곳: 환경변수 SO101_DIR, {', '.join(map(str, cands))}\n"
        f"  해결: python {SIM_DIR / 'fetch_so101.py'}   (약 19MB, Apache-2.0)\n"
        "        또는 이미 받은 폴더를 SO101_DIR 환경변수로 지정")


def fetch_target() -> Path:
    """fetch_so101.py 기본 저장 위치."""
    lr = local_root()
    return lr / "third_party" / "so101" if lr is not None else SIM_DIR / "third_party" / "so101"


def outputs_dir() -> Path:
    """영상·스틸·XML 기본 저장 폴더 (만들지는 않음)."""
    if os.environ.get("SIM_OUTPUTS"):
        return Path(os.environ["SIM_OUTPUTS"]).expanduser()
    lr = local_root()
    return lr / "outputs" / "sim" if lr is not None else SIM_DIR / "outputs"
