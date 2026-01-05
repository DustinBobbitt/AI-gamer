from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_dir(base: str | Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    run_dir = base_path / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dump_json(path: str | Path, data: Any) -> None:
    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=default)


def dump_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=default) + os.linesep)


def dump_text(path: str | Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

