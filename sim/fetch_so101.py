"""공식 SO-101 MuJoCo/URDF 파일을 받는다 (약 19MB, Apache-2.0). 이미 있는 파일은 건너뛴다.

출처: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
깃 저장소에는 넣지 않고 이 스크립트로 재생성한다.

기본 저장 위치 (sim_paths.fetch_target()):
  1. 환경변수 UNITA_LOCAL 이 있으면  $UNITA_LOCAL/third_party/so101
  2. 이 폴더의 ../../local 이 있으면  그 아래 third_party/so101 (또는 ~/UNITA_PAC2026/local)
  3. 아니면                          <sim>/third_party/so101 (sim/.gitignore 로 제외됨)

  python fetch_so101.py              # 기본 위치
  python fetch_so101.py --dest DIR   # 직접 지정 (그다음 SO101_DIR=DIR 로 실행)
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_paths import fetch_target  # noqa: E402

REPO, REF, SUB = "TheRobotStudio/SO-ARM100", "main", "Simulation/SO101/"


def fetch(dest: Path) -> Path:
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1") as r:
        tree = json.load(r)["tree"]
    files = [
        x["path"]
        for x in tree
        if x["type"] == "blob" and x["path"].startswith(SUB) and not x["path"].endswith(".part")
    ]
    for p in files:
        dst = dest / p[len(SUB) :]
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/{REF}/{p}", dst)
        print("받음", dst.relative_to(dest))
    print(f"{len(files)}개 파일 → {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(description="SO-101 공식 MJCF/STL 받기")
    ap.add_argument("--dest", type=Path, default=None, help=f"저장 폴더 (기본: {fetch_target()})")
    a = ap.parse_args()
    dest = fetch(a.dest or fetch_target())
    if a.dest is not None:
        print(f"이 위치는 자동 탐색 경로가 아닐 수 있습니다 → export SO101_DIR={dest}")


if __name__ == "__main__":
    main()
