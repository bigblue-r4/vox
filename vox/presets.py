"""Preset schema, bounded parameter ranges and sampling.

A preset is a small, fully serialisable description of a voice "color": a
chain of pitch/formant/EQ/saturation/modulation stages. Ranges are bounded on
purpose — the generator draws only from ``PARAMS`` and ``validate`` enforces
the same table, so no preset can wander into territory the intelligibility
guard has never been calibrated against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "vox.preset/1"

#: name -> (min, max, sampling)
#: ``log`` is used for frequency-like parameters so the draw is perceptually flat.
#:
#: The bounds are the guard-passing region: they were set from the measured
#: STI/WER response of each parameter (see README "Bound calibration") so that
#: a random draw inside them passes the 5% guard most of the time, while still
#: allowing tail combinations the generator must reject.
PARAMS: dict[str, tuple[float, float, str]] = {
    "pitch_semitones": (-3.5, 3.5, "linear"),
    "formant_shift": (0.90, 1.13, "linear"),
    "low_shelf_db": (-6.0, 6.0, "linear"),
    "high_shelf_db": (-9.0, 7.0, "linear"),
    "peak_freq": (250.0, 7000.0, "log"),
    "peak_db": (-6.0, 6.0, "linear"),
    "peak_q": (0.6, 2.4, "linear"),
    "drive": (0.0, 0.38, "linear"),
    "ring_amount": (0.0, 0.20, "linear"),
    "ring_hz": (40.0, 900.0, "log"),
    "tremolo_hz": (0.0, 8.0, "linear"),
    "tremolo_depth": (0.0, 0.38, "linear"),
    "comb_delay_ms": (3.0, 18.0, "linear"),
    "comb_mix": (0.0, 0.30, "linear"),
    "crush_bits": (9.0, 16.0, "linear"),
    "hardness": (0.0, 0.65, "linear"),
    "shimmer": (0.0, 0.35, "linear"),
    "shimmer_hz": (4.0, 12.0, "log"),
    "noise_mix": (0.0, 0.22, "linear"),
    "noise_colour": (0.0, 1.0, "linear"),
    "output_gain_db": (-4.0, 4.0, "linear"),
}

#: A signature is a stylistic archetype: it pins a few parameters to a
#: characteristic sub-range, and the remaining "color" parameters are drawn as
#: small jitter around neutral so the result reads as one deliberate colour.
SIGNATURES: dict[str, dict[str, tuple[float, float]]] = {
    "air": {
        "high_shelf_db": (2.0, 7.0),
        "noise_mix": (0.015, 0.05),
        "noise_colour": (0.55, 0.9),
        "peak_freq": (3500.0, 7000.0),
        "peak_db": (0.0, 3.0),
    },
    "warm": {
        "low_shelf_db": (2.5, 6.0),
        "high_shelf_db": (-7.0, -1.5),
        "drive": (0.05, 0.22),
    },
    "glass": {
        "peak_freq": (3200.0, 6200.0),
        "peak_db": (2.5, 6.0),
        "formant_shift": (1.03, 1.12),
        "high_shelf_db": (1.0, 4.0),
        "shimmer": (0.04, 0.12),
    },
    "shimmer": {
        "shimmer": (0.12, 0.30),
        "shimmer_hz": (6.0, 11.0),
        "high_shelf_db": (0.5, 3.5),
        "noise_mix": (0.01, 0.04),
        "noise_colour": (0.6, 0.95),
    },
    "hiss": {
        "noise_mix": (0.07, 0.16),
        "noise_colour": (0.7, 1.0),
        "hardness": (0.2, 0.45),
        "crush_bits": (11.0, 14.0),
    },
    "hard": {
        "hardness": (0.25, 0.55),
        "crush_bits": (10.0, 13.0),
        "peak_db": (1.0, 3.0),
        "high_shelf_db": (0.0, 3.0),
    },
    "velvet": {
        "low_shelf_db": (1.5, 5.0),
        "high_shelf_db": (-4.0, 0.0),
        "peak_db": (-3.0, 0.5),
        "comb_mix": (0.05, 0.18),
    },
    "smear": {
        "comb_mix": (0.14, 0.30),
        "comb_delay_ms": (9.0, 18.0),
        "formant_shift": (0.96, 1.06),
    },
    "shimmer": {
        "tremolo_depth": (0.12, 0.35),
        "tremolo_hz": (2.5, 7.0),
        "high_shelf_db": (0.5, 4.0),
    },
    "grit": {
        "drive": (0.18, 0.36),
        "crush_bits": (9.0, 13.0),
        "peak_db": (-2.0, 2.5),
    },
    "close": {
        "pitch_semitones": (-3.0, -1.2),
        "formant_shift": (0.92, 0.99),
        "low_shelf_db": (2.0, 5.0),
    },
    "bright": {
        "pitch_semitones": (1.0, 3.0),
        "formant_shift": (1.0, 1.09),
        "high_shelf_db": (1.5, 5.0),
    },
    "static": {
        "ring_amount": (0.05, 0.18),
        "ring_hz": (60.0, 900.0),
        "peak_db": (-2.0, 2.0),
    },
    # No pinned parameters: every parameter is drawn across its full bounded
    # range. Wildcards are the generator's tail exploration, so they are the
    # draws most likely to be refused by the guard.
    "wildcard": {},
}

#: Neutrally transparent chain; signatures and jitter are applied on top.
NEUTRAL: dict[str, float] = {
    "pitch_semitones": 0.0,
    "formant_shift": 1.0,
    "low_shelf_db": 0.0,
    "high_shelf_db": 0.0,
    "peak_freq": 1500.0,
    "peak_db": 0.0,
    "peak_q": 1.0,
    "drive": 0.0,
    "ring_amount": 0.0,
    "ring_hz": 250.0,
    "tremolo_hz": 1.0,
    "tremolo_depth": 0.0,
    "comb_delay_ms": 8.0,
    "comb_mix": 0.0,
    "crush_bits": 16.0,
    "hardness": 0.0,
    "shimmer": 0.0,
    "shimmer_hz": 6.5,
    "noise_mix": 0.0,
    "noise_colour": 0.5,
    "output_gain_db": 0.0,
}

#: Half-width of the "color" jitter around neutral, per parameter.
_JITTER: dict[str, float] = {
    "pitch_semitones": 2.0,
    "formant_shift": 0.07,
    "low_shelf_db": 3.0,
    "high_shelf_db": 3.5,
    "peak_freq": 0.8,  # octaves
    "peak_db": 2.5,
    "peak_q": 0.4,
    "drive": 0.12,
    "ring_amount": 0.07,
    "ring_hz": 0.8,  # octaves
    "tremolo_hz": 2.5,
    "tremolo_depth": 0.14,
    "comb_delay_ms": 4.0,
    "comb_mix": 0.12,
    "crush_bits": 3.0,
    "hardness": 0.3,
    "shimmer": 0.1,
    "shimmer_hz": 0.6,  # log
    "noise_mix": 0.03,
    "noise_colour": 0.35,
    "output_gain_db": 2.5,
}

#: Parameters sampled as a rounded integer.
INT_PARAMS = {"crush_bits"}

#: Absolute tightest chain envelope the guard was calibrated against.
MAX_PARAM_COUNT = len(PARAMS)


class PresetError(ValueError):
    """Raised for malformed or out-of-range presets."""


@dataclass
class Preset:
    name: str
    params: dict[str, float]
    source: str | None = None
    seed: dict[str, int] | None = None
    notes: str = ""
    origin: str = "generated"
    signature: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "origin": self.origin,
            "signature": self.signature,
            "source": self.source,
            "seed": self.seed,
            "notes": self.notes,
            "params": round_params(self.params),
            **self.extra,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "Preset":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PresetError(f"{path}: invalid JSON ({exc})") from exc
        return cls.from_dict(raw, where=str(path))

    @classmethod
    def from_dict(cls, raw: dict, where: str = "<dict>") -> "Preset":
        if not isinstance(raw, dict):
            raise PresetError(f"{where}: preset must be a JSON object")
        if raw.get("schema") not in (None, SCHEMA):
            raise PresetError(f"{where}: unsupported schema {raw.get('schema')!r}")
        params = raw.get("params")
        if not isinstance(params, dict):
            raise PresetError(f"{where}: missing 'params' object")
        unknown = set(params) - set(PARAMS)
        if unknown:
            raise PresetError(f"{where}: unknown parameters {sorted(unknown)}")
        missing = set(PARAMS) - set(params)
        if missing:
            raise PresetError(f"{where}: missing parameters {sorted(missing)}")
        clean = {k: float(v) for k, v in params.items()}
        validate(clean, where=where)
        name = str(raw.get("name") or Path(where).stem or "preset")
        return cls(
            name=name,
            params=clean,
            source=raw.get("source"),
            seed=raw.get("seed"),
            notes=raw.get("notes") or "",
            origin=raw.get("origin") or "generated",
            signature=raw.get("signature"),
            extra={
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "schema",
                    "name",
                    "origin",
                    "signature",
                    "source",
                    "seed",
                    "notes",
                    "params",
                }
            },
        )


def _round(v: float) -> float:
    return float(round(float(v), 4))


def round_params(params: dict[str, float]) -> dict[str, float]:
    """Round to the precision written to preset/meta files."""
    return {k: _round(v) for k, v in params.items()}


def validate(params: dict[str, float], where: str = "<preset>") -> None:
    for key, (lo, hi, _mode) in PARAMS.items():
        if key not in params:
            raise PresetError(f"{where}: missing {key!r}")
        v = float(params[key])
        if not np.isfinite(v):
            raise PresetError(f"{where}: {key} is not finite")
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            raise PresetError(f"{where}: {key}={v:g} outside [{lo:g}, {hi:g}]")


def _clip(key: str, value: float) -> float:
    lo, hi, _ = PARAMS[key]
    return float(min(hi, max(lo, value)))


def draw_colors(rng: np.random.Generator) -> tuple[str, dict[str, float]]:
    """Draw a signature and its color parameters.

    The signature pins its own parameters; everything else gets a small
    jitter around neutral. Every value is clipped into ``PARAMS`` so the
    result always validates.
    """
    # One draw in five is a wildcard, so the guard's reject-and-regenerate
    # path is exercised by ordinary runs instead of only in theory.
    if rng.random() < 0.2:
        signature = "wildcard"
    else:
        signature = str(rng.choice([s for s in sorted(SIGNATURES) if s != "wildcard"]))
    pinned = SIGNATURES[signature]
    out = dict(NEUTRAL)

    if signature == "wildcard":
        # Tail exploration: half of the parameters are pushed to the outer 25%
        # of their range, so combinations that the guard should refuse actually
        # occur and get regenerated. Values always stay inside PARAMS.
        for key, (lo, hi, mode) in PARAMS.items():
            edge = rng.random() < 0.5
            high_side = rng.random() < 0.5
            if mode == "log":
                a, b = float(np.log(lo)), float(np.log(hi))
                if edge:
                    span = (b - a) * 0.25
                    a, b = (b - span, b) if high_side else (a, a + span)
                value = float(np.exp(rng.uniform(a, b)))
            else:
                a, b = lo, hi
                if edge:
                    span = (hi - lo) * 0.25
                    a, b = (hi - span, hi) if high_side else (lo, lo + span)
                value = float(rng.uniform(a, b))
            if key in INT_PARAMS:
                value = float(int(round(value)))
            out[key] = _clip(key, value)
        return signature, out

    for key, half_width in _JITTER.items():
        if key in pinned:
            continue
        if PARAMS[key][2] == "log":
            out[key] = float(np.exp(np.log(NEUTRAL[key]) + rng.uniform(-half_width, half_width)))
        else:
            out[key] = NEUTRAL[key] + float(rng.uniform(-half_width, half_width))

    for key, (lo, hi) in pinned.items():
        if PARAMS[key][2] == "log":
            out[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[key] = float(rng.uniform(lo, hi))

    for key in out:
        if key in INT_PARAMS:
            out[key] = float(int(round(out[key])))
        out[key] = _clip(key, out[key])
    return signature, out


def draw_params(rng: np.random.Generator) -> dict[str, float]:
    """Draw one preset from the bounded ranges (signature discarded)."""
    return draw_colors(rng)[1]


def encode(params: dict[str, float]) -> str:
    """Compact one-line summary for shortlists and logs."""
    p = params
    bits = [
        f"pit {p['pitch_semitones']:+.1f}st",
        f"fmt {p['formant_shift']:.2f}x",
        f"eq {p['low_shelf_db']:+.0f}/{p['peak_db']:+.0f}@{p['peak_freq'] / 1000:.1f}k/{p['high_shelf_db']:+.0f}dB",
    ]
    if p["drive"] > 0.01:
        bits.append(f"drive {p['drive']:.2f}")
    if p["ring_amount"] > 0.01:
        bits.append(f"ring {p['ring_hz']:.0f}Hz {p['ring_amount']:.2f}")
    if p["tremolo_depth"] > 0.01 and p["tremolo_hz"] > 0.01:
        bits.append(f"trem {p['tremolo_hz']:.1f}Hz {p['tremolo_depth']:.2f}")
    if p["comb_mix"] > 0.01:
        bits.append(f"comb {p['comb_delay_ms']:.1f}ms {p['comb_mix']:.2f}")
    if p["crush_bits"] < 15.5:
        bits.append(f"crush {int(p['crush_bits'])}b")
    if p["hardness"] > 0.02:
        bits.append(f"hard {p['hardness']:.2f}")
    if p["shimmer"] > 0.02:
        bits.append(f"shim {p['shimmer']:.2f}@{p['shimmer_hz']:.1f}Hz")
    if p["noise_mix"] > 0.005:
        bits.append(f"noise {p['noise_mix']:.2f}/{p['noise_colour']:.1f}")
    return " · ".join(bits)


def load_dir(directory) -> list[Preset]:
    """Load every preset under ``directory``, including ``<name>/preset.json``
    layouts used by ``presets/shipped/``."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.rglob("*.json")):
        if path.name in {"index.json", "meta.json"}:
            continue
        # nested layouts must use <name>/preset.json; only the directory's own
        # files are presets otherwise (presets/designed/*.json)
        if path.parent != directory and path.name != "preset.json":
            continue
        found.append(Preset.load(path))
    return found
