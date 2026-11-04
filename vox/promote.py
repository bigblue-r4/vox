"""Promote kept presets from a run into ``presets/shipped/``.

The shortlist is the decision surface; this module is the only thing that
writes the shipped set. Promotion copies the preset, both audio files and the
metadata, then maintains ``presets/shipped/index.json`` with origin, seed and
guard numbers so every shipped preset can be traced back to its run.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, presets

SHIPPED_DIR = Path("presets/shipped")
DESIGNED_DIR = Path("presets/designed")
INDEX_NAME = "index.json"
TARGET_MIN = 6
TARGET_MAX = 8


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_index(shipped_dir: Path) -> dict:
    path = shipped_dir / INDEX_NAME
    if not path.exists():
        return {"schema": "vox.shipped/1", "updated_at": None, "presets": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(shipped_dir: Path, index: dict) -> Path:
    index["updated_at"] = _now()
    index["tool_version"] = __version__
    index["target"] = {"min": TARGET_MIN, "max": TARGET_MAX}
    # "curated" presets are generated draws that were tuned by ear before
    # shipping; they count against the same 6-8 target as raw generated ones.
    index["counts"] = {
        "generated": sum(
            1 for p in index["presets"] if p["origin"] in ("generated", "curated")
        ),
        "curated": sum(1 for p in index["presets"] if p["origin"] == "curated"),
        "designed": sum(1 for p in index["presets"] if p["origin"] == "designed"),
        "total": len(index["presets"]),
    }
    path = shipped_dir / INDEX_NAME
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return path


def _run_item(run_dir: Path, name: str) -> tuple[dict, Path]:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index["items"]:
        if entry["name"] == name:
            return entry, run_dir / entry["dir"]
    raise KeyError(f"{name!r} is not part of run {run_dir}")


def promote(
    run_dir: str | Path,
    names: list[str],
    shipped_dir: str | Path = SHIPPED_DIR,
    force: bool = False,
) -> list[str]:
    """Copy ``names`` from ``run_dir`` into the shipped set.

    Idempotent: promoting the same preset twice does not duplicate the index
    entry. A name collision with different parameters requires ``force``.
    """
    run_dir = Path(run_dir)
    shipped_dir = Path(shipped_dir)
    shipped_dir.mkdir(parents=True, exist_ok=True)
    index = _load_index(shipped_dir)
    promoted: list[str] = []

    for name in names:
        entry, item_dir = _run_item(run_dir, name)
        if not (item_dir / "preset.json").exists():
            raise FileNotFoundError(f"{item_dir}/preset.json missing")
        target = shipped_dir / name
        existing = next((p for p in index["presets"] if p["name"] == name), None)
        if existing is not None and existing["params"] != entry["params"] and not force:
            raise FileExistsError(
                f"shipped preset {name!r} exists with different parameters (use --force)"
            )
        target.mkdir(parents=True, exist_ok=True)
        preset_path = item_dir / "preset.json"
        preset = presets.Preset.load(preset_path)
        preset.origin = "generated"
        preset.save(target / "preset.json")
        for fname in ("sample.wav", "audition.wav", "meta.json"):
            src = item_dir / fname
            if src.exists():
                shutil.copy2(src, target / fname)

        record = {
            "name": name,
            "origin": "generated",
            "signature": entry.get("signature"),
            "source": entry.get("source"),
            "params": entry["params"],
            "guard": entry["guard"],
            "run": str(run_dir),
            "seed": {"master": json.loads((run_dir / "index.json").read_text(encoding="utf-8"))["master_seed"], "index": entry["index"]},
            "promoted_at": _now(),
            "dir": name,
        }
        if existing is None:
            index["presets"].append(record)
        else:
            index["presets"] = [
                record if p["name"] == name else p for p in index["presets"]
            ]
        promoted.append(name)

    _write_index(shipped_dir, index)
    return promoted


def demote(names: list[str], shipped_dir: str | Path = SHIPPED_DIR) -> list[str]:
    shipped_dir = Path(shipped_dir)
    index = _load_index(shipped_dir)
    removed = []
    for name in names:
        if any(p["name"] == name for p in index["presets"]):
            index["presets"] = [p for p in index["presets"] if p["name"] != name]
            target = shipped_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            removed.append(name)
    _write_index(shipped_dir, index)
    return removed


def register_designed(designed_dir: str | Path = DESIGNED_DIR,
                      shipped_dir: str | Path = SHIPPED_DIR) -> list[str]:
    """List the hand-designed presets in the shipped index (they live in
    ``presets/designed/`` and are referenced, not copied)."""
    designed_dir, shipped_dir = Path(designed_dir), Path(shipped_dir)
    shipped_dir.mkdir(parents=True, exist_ok=True)
    index = _load_index(shipped_dir)
    names = []
    for preset_path in sorted(designed_dir.glob("*.json")):
        preset = presets.Preset.load(preset_path)
        names.append(preset.name)
        record = {
            "name": preset.name,
            "origin": "designed",
            "signature": preset.signature or "hand-designed",
            "source": None,
            "params": presets.round_params(preset.params),
            "guard": None,
            "run": None,
            "seed": None,
            "promoted_at": None,
            "dir": f"../designed/{preset_path.name}",
        }
        existing = next((p for p in index["presets"] if p["name"] == preset.name), None)
        if existing is None:
            index["presets"].append(record)
        else:
            index["presets"] = [
                record if p["name"] == preset.name else p for p in index["presets"]
            ]
    _write_index(shipped_dir, index)
    return names


def status(shipped_dir: str | Path = SHIPPED_DIR) -> dict:
    index = _load_index(Path(shipped_dir))
    generated = [
        p for p in index["presets"] if p["origin"] in ("generated", "curated")
    ]
    curated = [p for p in index["presets"] if p["origin"] == "curated"]
    designed = [p for p in index["presets"] if p["origin"] == "designed"]
    if len(generated) < TARGET_MIN:
        state = "under"
    elif len(generated) > TARGET_MAX:
        state = "over"
    else:
        state = "ok"
    return {
        "generated": len(generated),
        "curated": len(curated),
        "designed": len(designed),
        "total": len(index["presets"]),
        "target_min": TARGET_MIN,
        "target_max": TARGET_MAX,
        "state": state,
        "names": [p["name"] for p in index["presets"]],
    }
