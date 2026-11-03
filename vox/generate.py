"""Addendum 2 — generated default presets.

``vox generate --count 12 --seed 404``:

* procedurally synthesises one original colour sample per preset (six sources,
  balanced round-robin),
* draws a signature + colour parameters from the bounded ranges,
* renders the standard test phrase through the preset, estimates WER and
  *regenerates* the preset while the estimate is above the 5% guard,
* writes ``sample.wav``, ``audition.wav``, ``preset.json`` and ``meta.json`` per
  preset plus ``seeds.jsonl``, ``reference.wav``, ``index.json`` and
  ``shortlist.html``,
* logs every attempt (accepted and rejected) so any preset is exactly
  reproducible from its master seed and item index.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import TEST_PHRASE, __version__, color, dsp, intelligibility as ig, presets, synth, testphrase

RUN_SCHEMA = "vox.run/1"

NAME_STEMS = (
    "ember", "cinder", "halcyon", "quartz", "morrow", "vellum", "lumen",
    "aster", "sable", "harbor", "willow", "orchid", "nimbus", "cobalt",
    "saffron", "meridian", "cascade", "atlas", "ivory", "umber", "juniper",
    "lantern", "tidal", "fresnel", "obsidian", "marigold", "pewter",
    "solstice", "gossamer", "kestrel",
)


@dataclass
class Item:
    name: str
    index: int
    source: str
    signature: str
    params: dict[str, float]
    sti: float
    wer: float
    attempt: int

    def to_index_entry(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "source": self.source,
            "signature": self.signature,
            "params": presets.round_params(self.params),
            "guard": {"sti": round(self.sti, 4), "est_wer": round(self.wer, 4)},
            "dir": f"items/{self.name}",
        }


@dataclass
class RunResult:
    run_dir: Path
    master_seed: int
    items: list[Item] = field(default_factory=list)
    rejected: int = 0
    index_path: Path | None = None
    shortlist_path: Path | None = None

    @property
    def accepted(self) -> int:
        return len(self.items)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _unique_name(rng: np.random.Generator, used: set[str]) -> str:
    for _ in range(200):
        stem = str(rng.choice(NAME_STEMS))
        number = int(rng.integers(1, 100))
        candidate = f"{stem}-{number:02d}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError("could not find a unique preset name")


def phrase_reference(sr: int = dsp.SR) -> np.ndarray:
    """The clean carrier used by the guard and for dry auditions."""
    return testphrase.synthesize(TEST_PHRASE, sr=sr)


def _build_item(
    master_seed: int,
    index: int,
    used_names: set[str],
    reference: np.ndarray,
    guard: float,
    max_attempts: int,
    sr: int,
) -> tuple[Item, list[dict], np.ndarray, np.ndarray]:
    """Generate one accepted item: (item, attempt log, sample, audition).

    Deterministic: the RNG is derived from ``SeedSequence([master_seed, index])``
    and every draw happens in a fixed order, so replaying the seed reproduces
    the parameters *and* the audio exactly.
    """
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, index]))
    source = synth.source_for_index(index)
    name = _unique_name(rng, used_names)
    sample = synth.synthesize(source, rng)
    if sample.size < sr:  # defensive: every source renders >= 2 s
        raise RuntimeError(f"source {source!r} produced an unexpectedly short sample")

    log: list[dict] = []
    for attempt in range(1, max_attempts + 1):
        signature, params = presets.draw_colors(rng)
        audition = dsp.loudness_match(color.render(reference, params, sr, rng=rng), reference)
        assessment = ig.assess(reference, audition)
        accepted = assessment.wer <= guard
        log.append(
            {
                "master_seed": master_seed,
                "index": index,
                "name": name,
                "source": source,
                "signature": signature,
                "attempt": attempt,
                "status": "accepted" if accepted else "rejected",
                "guard": {
                    "sti": round(assessment.sti, 6),
                    "est_wer": round(assessment.wer, 6),
                    "threshold": guard,
                },
                "params": presets.round_params(params),
            }
        )
        if accepted:
            item = Item(
                name=name,
                index=index,
                source=source,
                signature=signature,
                params=params,
                sti=assessment.sti,
                wer=assessment.wer,
                attempt=attempt,
            )
            return item, log, sample, audition
    raise RuntimeError(
        f"item {index}: no preset passed the {guard:.0%} guard in {max_attempts} attempts"
    )


def generate(
    count: int = 12,
    seed: int | None = None,
    out_root: str | Path = "runs",
    guard: float = ig.WER_THRESHOLD,
    max_attempts: int = 30,
    sr: int = dsp.SR,
    overwrite: bool = False,
    progress=None,
) -> RunResult:
    if count < 1:
        raise ValueError("--count must be >= 1")
    master_seed = int(np.random.SeedSequence().entropy % (2**31)) if seed is None else int(seed)

    run_dir = Path(out_root) / f"seed-{master_seed:06d}"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists (use --force to overwrite)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    reference = phrase_reference(sr)
    reference_path = dsp.write_wav(run_dir / "reference.wav", reference, sr)

    used: set[str] = set()
    items: list[Item] = []
    seeds_lines: list[dict] = []
    for i in range(count):
        item, log, sample, audition = _build_item(
            master_seed, i, used, reference, guard, max_attempts, sr
        )
        seeds_lines.extend(log)
        item_dir = run_dir / "items" / item.name
        item_dir.mkdir(parents=True, exist_ok=True)

        sample_path = dsp.write_wav(item_dir / "sample.wav", sample, sr)
        audition_path = dsp.write_wav(item_dir / "audition.wav", audition, sr)

        preset = presets.Preset(
            name=item.name,
            params=item.params,
            source=item.source,
            seed={"master": master_seed, "index": i},
            signature=item.signature,
            notes=f"generated · {synth.SOURCE_BLURB[item.source]}",
        )
        preset.save(item_dir / "preset.json")

        meta = {
            "name": item.name,
            "created_at": _now(),
            "tool_version": __version__,
            "seed": {"master": master_seed, "index": i},
            "attempt": item.attempt,
            "source": item.source,
            "signature": item.signature,
            "params": presets.round_params(item.params),
            "guard": {
                "sti": round(item.sti, 6),
                "est_wer": round(item.wer, 6),
                "threshold": guard,
                "passes": True,
            },
            "phrase": TEST_PHRASE,
            "files": {"sample": "sample.wav", "audition": "audition.wav"},
            "hashes": {
                "sample_sha256": dsp.sha256_file(sample_path),
                "audition_sha256": dsp.sha256_file(audition_path),
            },
            "audio": {
                "sample_duration_s": round(sample.size / sr, 3),
                "audition_duration_s": round(audition.size / sr, 3),
                "sample_peak_dbfs": round(float(dsp.gain_to_db(dsp.peak(sample))), 2),
                "audition_rms_dbfs": round(float(dsp.gain_to_db(dsp.rms(audition))), 2),
            },
        }
        (item_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        items.append(item)
        if progress:
            progress(item, len(log))

    with open(run_dir / "seeds.jsonl", "w", encoding="utf-8") as fh:
        for line in seeds_lines:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    run_index = {
        "schema": RUN_SCHEMA,
        "tool_version": __version__,
        "created_at": _now(),
        "master_seed": master_seed,
        "count": count,
        "max_attempts": max_attempts,
        "phrase": TEST_PHRASE,
        "guard": {
            "threshold": guard,
            "sti_midpoint": ig.STI_MIDPOINT,
            "logistic_k": ig.K,
        },
        "reference": {"path": "reference.wav", "sha256": dsp.sha256_file(reference_path)},
        "stats": {
            "accepted": len(items),
            "rejected": sum(1 for line in seeds_lines if line["status"] == "rejected"),
        },
        "items": [item.to_index_entry() for item in items],
    }
    index_path = run_dir / "index.json"
    index_path.write_text(json.dumps(run_index, indent=2) + "\n", encoding="utf-8")

    tags_path = run_dir / "tags.json"
    if not tags_path.exists():
        tags_path.write_text("{}\n", encoding="utf-8")

    from . import shortlist  # local import: shortlist imports promote

    shortlist_path = shortlist.write_shortlist(run_dir, run_index)

    return RunResult(
        run_dir=run_dir,
        master_seed=master_seed,
        items=items,
        rejected=run_index["stats"]["rejected"],
        index_path=index_path,
        shortlist_path=shortlist_path,
    )


def load_run(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (is this a vox run directory?)")
    return json.loads(path.read_text(encoding="utf-8"))


def read_seeds(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "seeds.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(run_dir: str | Path, sr: int = dsp.SR) -> tuple[bool, list[str]]:
    """Replay a run from its logged seeds and compare params and audio hashes."""
    run_dir = Path(run_dir)
    index = load_run(run_dir)
    reference = phrase_reference(sr)
    problems: list[str] = []
    used: set[str] = set()
    max_attempts = int(index.get("max_attempts", 30))
    for entry in index["items"]:
        i = int(entry["index"])
        try:
            item, _, sample, audition = _build_item(
                int(index["master_seed"]),
                i,
                used,
                reference,
                float(index["guard"]["threshold"]),
                max_attempts,
                sr,
            )
        except RuntimeError as exc:
            problems.append(f"{entry['name']}: replay failed ({exc})")
            continue
        if item.name != entry["name"]:
            problems.append(f"index {i}: replayed name {item.name!r} != {entry['name']!r}")
            continue
        if presets.round_params(item.params) != entry["params"]:
            problems.append(f"{entry['name']}: parameters differ on replay")
            continue
        item_dir = run_dir / entry["dir"]
        if dsp.hash_bytes(dsp.wav_file_bytes(sample, sr)) != dsp.sha256_file(item_dir / "sample.wav"):
            problems.append(f"{entry['name']}: sample.wav does not reproduce")
        if dsp.hash_bytes(dsp.wav_file_bytes(audition, sr)) != dsp.sha256_file(item_dir / "audition.wav"):
            problems.append(f"{entry['name']}: audition.wav does not reproduce")
    return (not problems, problems)
