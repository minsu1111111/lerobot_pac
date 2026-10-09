"""TTS 고정 문구. 키는 rollout/stt 쪽에서 그대로 쓰므로 바꾸지 않는다.

문구를 고치면 generate_wavs.py 를 다시 돌려 wavs/ 를 갱신하고 함께 커밋한다.

다른 프로젝트에서 다른 문장을 쓰려면 이 파일을 고치지 말고 문구 파일을 따로 만든다
(load_phrases 참고, README "다른 프로젝트에 넣기").
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PHRASES: dict[str, str] = {
    "ready": "준비됐습니다. 말씀해 주세요.",
    "start": "물을 따르겠습니다.",
    "done": "목표량까지 따랐습니다.",
    "placed": "컵을 놓았습니다.",
    "stopped": "정지했습니다.",
    "resume": "이어서 따르겠습니다.",
    "going_back": "원래 자리로 돌아가겠습니다.",
    "home": "초기 자세로 돌아가겠습니다.",
    "returned": "원래 자리로 돌아왔습니다.",
    "retry": "잘 못 들었어요. 다시 말씀해 주세요.",
    "timeout": "시간이 초과되어 동작을 마칩니다.",
}

# 큐가 꽉 차거나 정리될 때도 버리면 안 되는 문구
CRITICAL_KEYS = frozenset({"done", "placed", "stopped", "returned", "timeout"})

MANIFEST = "phrases.json"  # generate_wavs.py --phrases 가 wav 폴더에 함께 쓰는 문구 목록


def load_phrases(path: str | Path) -> tuple[dict[str, str], frozenset[str]]:
    """문구 파일 → (phrases, critical_keys).

    .json : {"phrases": {key: text}, "critical": [key, ...]}  또는  {key: text} (critical 없음)
    .py   : PHRASES = {...}, (선택) CRITICAL_KEYS = {...}
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("phrases"), dict):
            phrases, critical = data["phrases"], data.get("critical", [])
        else:
            phrases, critical = data, []
    elif path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(f"_tts_phrases_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "PHRASES"):
            raise ValueError(f"{path} 에 PHRASES 딕셔너리가 없습니다")
        phrases, critical = mod.PHRASES, getattr(mod, "CRITICAL_KEYS", [])
    else:
        raise ValueError(f"문구 파일은 .json 또는 .py 여야 합니다: {path}")
    if not isinstance(phrases, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in phrases.items()
    ):
        raise ValueError(f"{path}: 문구는 {{키: 문장}} 문자열 딕셔너리여야 합니다")
    bad = [k for k in phrases if not k or "/" in k or k.startswith(".")]
    if bad:
        raise ValueError(f"{path}: 파일 이름으로 쓸 수 없는 키 {bad}")
    return dict(phrases), frozenset(critical)


def save_manifest(wav_dir: str | Path, phrases: dict[str, str], critical=()) -> Path:
    path = Path(wav_dir) / MANIFEST
    path.write_text(
        json.dumps({"phrases": phrases, "critical": sorted(critical)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
