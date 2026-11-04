"""Write the shipped characters: the twelve presets' deliberate sound design.

The generated presets in ``presets/shipped/`` began as seeded draws, but each
one is now tuned to a named character so the set does not ship four variations
of the same colour. This script is the single place that table lives; running it
rewrites ``preset.json``, ``audition.wav`` and ``meta.json`` for every shipped
preset, rewrites the four designed presets, refreshes the shipped index and
reports each preset's guard estimate.

Run it after changing the chain, the parameters or the guard:

    .venv/bin/python tools/curate_shipped.py          # check only, no writes
    .venv/bin/python tools/curate_shipped.py --write   # apply
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vox import (  # noqa: E402
    TEST_PHRASE,
    __version__,
    color,
    dsp,
    generate,
    intelligibility as ig,
    presets,
    promote,
    synth,
)

N = presets.NEUTRAL


def mk(**overrides) -> dict[str, float]:
    params = dict(N)
    params.update(overrides)
    return params


#: name -> (character, source signature, params)
CHARACTERS: dict[str, tuple[str, str, dict]] = {
    "solstice-30": (
        "shimmering high halo: octave-up presence layer, fast 9.5 Hz shimmer, bright but soft-edged",
        "shimmer",
        mk(
            pitch_semitones=2.0, formant_shift=1.04, low_shelf_db=-1.0, high_shelf_db=3.5,
            peak_freq=4200.0, peak_db=1.5, peak_q=1.2, comb_mix=0.10, comb_delay_ms=9.0,
            hardness=0.05, shimmer=0.26, shimmer_hz=9.5, noise_mix=0.03, noise_colour=0.85,
        ),
    ),
    "ivory-18": (
        "white-noise texture with a hard edge: bright hiss bed, 13-bit crush, saturated presence",
        "hiss",
        mk(
            pitch_semitones=-1.2, formant_shift=0.97, low_shelf_db=1.5, high_shelf_db=2.5,
            peak_freq=3000.0, peak_db=1.0, drive=0.10, crush_bits=13.0, hardness=0.42,
            shimmer=0.05, shimmer_hz=7.0, noise_mix=0.14, noise_colour=0.95,
        ),
    ),
    "lumen-43": (
        "cavernous smear: long feedback comb, almost no EQ, tail-dominated",
        "smear",
        mk(
            pitch_semitones=-0.5, high_shelf_db=-1.0, comb_mix=0.30, comb_delay_ms=17.0,
            noise_mix=0.02, noise_colour=0.6, shimmer=0.02, shimmer_hz=6.0,
        ),
    ),
    "quartz-56": (
        "velvet: round low-mid, top rolled off, short comb bloom, no edge",
        "velvet",
        mk(
            pitch_semitones=-1.5, formant_shift=0.97, low_shelf_db=4.0, high_shelf_db=-4.0,
            peak_freq=700.0, peak_db=0.5, drive=0.06, comb_mix=0.15, comb_delay_ms=6.5,
            crush_bits=15.0, noise_mix=0.02, noise_colour=0.2,
        ),
    ),
    "sable-11": (
        "grit: driven and torn, 11-bit crush, presence peak at 3.2 kHz",
        "grit",
        mk(
            pitch_semitones=-0.8, peak_freq=3200.0, peak_db=2.5, high_shelf_db=1.0,
            drive=0.30, crush_bits=11.0, hardness=0.20, noise_mix=0.01, noise_colour=0.7,
        ),
    ),
    "harbor-02": (
        "close-mic intimacy: chesty low shelf, short room comb, dry",
        "close",
        mk(
            pitch_semitones=-2.6, formant_shift=0.94, low_shelf_db=4.0, high_shelf_db=1.0,
            peak_freq=750.0, peak_db=1.0, comb_mix=0.10, comb_delay_ms=11.0,
            tremolo_hz=2.5, tremolo_depth=0.03, crush_bits=15.0, noise_mix=0.02,
            noise_colour=0.35,
        ),
    ),
    "orchid-56": (
        "telephone band: both shelves cut hard, 1.6 kHz resonance, boxed",
        "static",
        mk(
            pitch_semitones=-1.0, formant_shift=1.02, low_shelf_db=-4.0, high_shelf_db=-7.0,
            peak_freq=1600.0, peak_db=2.2, peak_q=1.6, drive=0.05, crush_bits=15.0,
            noise_mix=0.01,
        ),
    ),
    "orchid-73": (
        "clean glass: minimal colour, gentle shimmer sheen, airy top",
        "glass",
        mk(
            pitch_semitones=-1.0, formant_shift=0.95, low_shelf_db=2.0, high_shelf_db=2.5,
            peak_freq=4000.0, peak_db=1.5, shimmer=0.08, shimmer_hz=7.0, noise_mix=0.02,
            noise_colour=0.8,
        ),
    ),
    "velvet-veil": (
        "breathy veil: dark soft noise under a low-passed voice",
        "hand-designed",
        mk(
            pitch_semitones=-1.0, formant_shift=0.97, low_shelf_db=2.0, high_shelf_db=-4.0,
            noise_mix=0.045, noise_colour=0.25, drive=0.05,
        ),
    ),
    "glasshouse": (
        "struck glass: 5.2 kHz bell, resonant comb tail, glassy shimmer",
        "hand-designed",
        mk(
            pitch_semitones=1.5, formant_shift=1.06, high_shelf_db=2.0, peak_freq=5200.0,
            peak_db=3.5, peak_q=1.2, shimmer=0.22, shimmer_hz=8.5, comb_mix=0.12,
            comb_delay_ms=12.0, drive=0.05, noise_mix=0.02, noise_colour=0.9,
        ),
    ),
    "static-bloom": (
        "broken radio: crushed, buzzing, hissy, unstable tremolo",
        "hand-designed",
        mk(
            high_shelf_db=-1.0, drive=0.28, crush_bits=9.0, noise_mix=0.10,
            noise_colour=0.5, tremolo_hz=2.4, tremolo_depth=0.20, hardness=0.35,
        ),
    ),
    "understudy": (
        "behind a door: top cut away, thin and distant, dulled",
        "hand-designed",
        mk(
            pitch_semitones=-1.0, formant_shift=0.965, low_shelf_db=-1.0,
            high_shelf_db=-5.0, peak_freq=900.0, peak_db=1.2, peak_q=0.8, drive=0.04,
            comb_mix=0.10, comb_delay_ms=9.0,
        ),
    ),
}

DESIGNED = {"velvet-veil", "glasshouse", "static-bloom", "understudy"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def assess(params: dict) -> ig.Assessment:
    phrase = generate.phrase_reference()
    out = dsp.loudness_match(color.render(phrase, params), phrase)
    return ig.assess(phrase, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply the changes")
    parser.add_argument("--shipped", default="presets/shipped")
    parser.add_argument("--designed", default="presets/designed")
    args = parser.parse_args(argv)

    shipped_dir, designed_dir = Path(args.shipped), Path(args.designed)
    phrase = generate.phrase_reference()
    results = {}
    problems = []

    for name, (character, signature, params) in CHARACTERS.items():
        presets.validate(params, where=name)
        out = dsp.loudness_match(
            color.render(phrase, params, rng=np.random.default_rng(20261104)), phrase
        )
        a = ig.assess(phrase, out)
        results[name] = (a, out, character, signature)
        state = "pass" if a.passes else "REFUSED"
        print(f"{name:<13} {signature:<14} STI {a.sti:.3f}  est WER {a.wer * 100:4.1f}%  {state}")
        if not a.passes:
            problems.append(name)

    if problems:
        print(f"\nnot writing: guard refuses {problems}")
        return 1
    if not args.write:
        print("\ncheck only (pass --write to apply)")
        return 0

    for name, (a, out, character, signature) in results.items():
        if name in DESIGNED:
            preset = presets.Preset(
                name=name,
                params=CHARACTERS[name][2],
                origin="designed",
                signature="hand-designed",
                notes=character,
            )
            preset.save(designed_dir / f"{name}.json")
            continue

        target = shipped_dir / name
        preset_path = target / "preset.json"
        if not preset_path.exists():
            raise FileNotFoundError(f"{preset_path} missing; cannot preserve sample provenance")
        # read the raw file: it may predate the current parameter table, so it
        # cannot be validated as a whole -- only source/seed provenance is kept.
        raw = json.loads(preset_path.read_text(encoding="utf-8"))
        preset = presets.Preset(
            name=name,
            params=CHARACTERS[name][2],
            source=raw.get("source"),
            seed=raw.get("seed"),
            origin="curated",
            signature=signature,
            notes=character,
        )
        preset.save(preset_path)

        audition_path = dsp.write_wav(target / "audition.wav", out)
        meta_path = target / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update(
            {
                "name": name,
                "character": character,
                "curated_at": _now(),
                "tool_version": __version__,
                "signature": signature,
                "origin": "curated",
                "params": presets.round_params(preset.params),
                "guard": {
                    "sti": round(a.sti, 6),
                    "est_wer": round(a.wer, 6),
                    "threshold": ig.WER_THRESHOLD,
                    "passes": True,
                },
                "hashes": {
                    "sample_sha256": dsp.sha256_file(target / "sample.wav"),
                    "audition_sha256": dsp.sha256_file(audition_path),
                },
                "audio": {
                    "sample_duration_s": round(
                        dsp.read_wav(target / "sample.wav")[0].size / dsp.SR, 3
                    ),
                    "audition_duration_s": round(out.size / dsp.SR, 3),
                    "sample_peak_dbfs": round(
                        float(dsp.gain_to_db(dsp.peak(dsp.read_wav(target / "sample.wav")[0]))), 2
                    ),
                    "audition_rms_dbfs": round(float(dsp.gain_to_db(dsp.rms(out))), 2),
                },
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    index = promote._load_index(shipped_dir)
    for record in index["presets"]:
        name = record["name"]
        a, _out, character, signature = results[name]
        record["params"] = presets.round_params(CHARACTERS[name][2])
        record["signature"] = signature
        record["character"] = character
        if name in DESIGNED:
            record["guard"] = {"sti": round(a.sti, 6), "est_wer": round(a.wer, 6)}
        else:
            record["origin"] = "curated"
            record["guard"] = {"sti": round(a.sti, 6), "est_wer": round(a.wer, 6)}
    promote._write_index(shipped_dir, index)

    (shipped_dir / "CHARACTERS.md").write_text(
        "# Shipped characters\n\n"
        "Each preset is tuned to one deliberate character so the set does not ship\n"
        "variations of a single colour. Regenerate with `tools/curate_shipped.py`.\n\n"
        + "\n".join(f"- **{name}** — {results[name][2]}" for name in sorted(results))
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(results)} presets + index ({shipped_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
