"""Sweep one parameter at a time and report the guard's estimated WER.

This is how the bounds in ``vox.presets.PARAMS`` were set: pick the largest
value whose estimate still passes with room to spare, then check a stacked
combination at the resulting tail. Run it after touching the chain or the
guard.

    .venv/bin/python tools/calibrate_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vox import color, dsp, generate, intelligibility as ig, presets  # noqa: E402

SWEEPS: dict[str, list[float]] = {
    "shimmer": [0.02, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.35],
    "hardness": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "noise_mix": [0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.22],
}


def score(phrase, **overrides) -> ig.Assessment:
    params = dict(presets.NEUTRAL)
    params.update(overrides)
    out = dsp.loudness_match(color.render(phrase, params), phrase)
    return ig.assess(phrase, out)


def main() -> int:
    phrase = generate.phrase_reference()

    base = score(phrase)
    print(f"neutral              STI {base.sti:.3f}  est WER {base.wer * 100:4.1f}%")

    for name, values in SWEEPS.items():
        print(f"\n{name}")
        for value in values:
            a = score(phrase, **{name: value})
            mark = "pass" if a.passes else "REFUSED"
            print(
                f"  {value:6.2f}            STI {a.sti:.3f}  "
                f"est WER {a.wer * 100:4.1f}%  {mark}"
            )

    print("\nnoise colour (mix 0.12)")
    for colour in (0.0, 0.25, 0.5, 0.75, 1.0):
        a = score(phrase, noise_mix=0.12, noise_colour=colour)
        print(
            f"  colour {colour:4.2f}       STI {a.sti:.3f}  "
            f"est WER {a.wer * 100:4.1f}%  {'pass' if a.passes else 'REFUSED'}"
        )

    print("\nshimmer rate (shimmer 0.20)")
    for hz in (4.0, 6.5, 9.0, 12.0):
        a = score(phrase, shimmer=0.20, shimmer_hz=hz)
        print(
            f"  {hz:5.1f} Hz          STI {a.sti:.3f}  "
            f"est WER {a.wer * 100:4.1f}%  {'pass' if a.passes else 'REFUSED'}"
        )

    combos = {
        "shimmer tail": {"shimmer": 0.30, "shimmer_hz": 10.0, "high_shelf_db": 3.0},
        "noise tail": {"noise_mix": 0.20, "noise_colour": 1.0, "hardness": 0.5},
        "hard tail": {"hardness": 0.9, "crush_bits": 10.0},
        "full stack": {
            "shimmer": 0.25,
            "hardness": 0.55,
            "noise_mix": 0.14,
            "noise_colour": 0.9,
            "pitch_semitones": 2.4,
            "crush_bits": 11.0,
        },
    }
    print("\ncombinations")
    for label, overrides in combos.items():
        a = score(phrase, **overrides)
        print(
            f"  {label:<18} STI {a.sti:.3f}  est WER {a.wer * 100:4.1f}%  "
            f"{'pass' if a.passes else 'REFUSED'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
