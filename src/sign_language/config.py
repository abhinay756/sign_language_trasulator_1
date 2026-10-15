from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "classes.json"
DATA_DIR = PROJECT_ROOT / "data" / "landmarks"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    classes = config.get("classes")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise ValueError("config/classes.json must contain a non-empty unique 'classes' list")
    sequence_length = int(config.get("sequence_length", 30))
    if sequence_length < 4:
        raise ValueError("sequence_length must be at least 4")
    config["sequence_length"] = sequence_length
    return config
