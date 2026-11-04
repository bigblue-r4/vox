"""Render and measure the three accent options over the same spoken turn.

The point of this tool is that "soft accent" stops being an adjective: every
option is mixed at a stated level below the voice and then measured with the
same intelligibility guard the presets use.

    .venv/bin/python tools/accent_demo.py --out runs/accent-demo
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vox import accent, dsp, intelligibility as ig, testphrase  # noqa: E402

SR = dsp.SR
LEAD = 0.30
TAIL = 0.55
GAP = 0.45

SENT_A = "the quick brown fox"
SENT_B = "jumps over the lazy dog"


def build_voice(sr: int = SR):
    """A two-sentence turn: this is what an ordinary answer looks like."""
    a = testphrase.synthesize(SENT_A, sr=sr)
    b = testphrase.synthesize(SENT_B, sr=sr)
    gap = np.zeros(int(round(GAP * sr)))
    voice = np.concatenate([a, gap, b])
    spans = [
        (0.0, a.size / sr),
        ((a.size + gap.size) / sr, voice.size / sr),
    ]
    return voice, spans


def padded(voice, lead: float = LEAD, tail: float = TAIL, sr: int = SR) -> np.ndarray:
    return np.concatenate([
        np.zeros(int(round(lead * sr))),
        voice,
        np.zeros(int(round(tail * sr))),
    ])


def cue_track(size: int, pieces, sr: int = SR) -> np.ndarray:
    """A silent-length track holding only the cues, for overlap measurement."""
    out = np.zeros(size)
    for at_s, cue in pieces:
        start = int(round(at_s * sr))
        end = min(size, start + cue.size)
        out[start:end] += cue[: end - start]
    return out


def option_chime_per_sentence(base, spans, level_db: float, sr: int = SR):
    """One bell at the end of every sentence -- the metronome case."""
    length = base.size / sr
    bell = accent.chime(0.9, 1046.5, sr)
    pieces = []
    for _, end_s in spans:
        # fires the instant the sentence ends, so the tail runs under the next
        pieces.append((min(end_s, length - 0.02), bell))
    out = base.copy()
    for at_s, cue in pieces:
        out = accent.place(out, cue, at_s=at_s, level_db=level_db, sr=sr)
    return out, pieces, len(pieces)


def option_hum_bed(base, spans, level_db: float, sr: int = SR):
    """A continuous low bed under the whole turn."""
    dur = base.size / sr
    bed = accent.hum(dur - 0.1, 55.0, sr)
    out = accent.place(base, bed, at_s=0.05, level_db=level_db, sr=sr)
    return out, [(0.05, bed)], 1


def option_turn_signature(base, spans, level_db: float, sr: int = SR):
    """One breath before the turn, one settle after it."""
    inhale = accent.breath(0.24, sr, np.random.default_rng(7))
    # ends ~60 ms before the first word
    tail_start = spans[-1][1] + 0.02
    body = accent.settle(0.60, sr)
    pieces = [(0.0, inhale), (tail_start, body)]
    out = accent.place(base, inhale, at_s=0.0, level_db=level_db, sr=sr)
    out = accent.place(out, body, at_s=tail_start, level_db=level_db - 2.0, sr=sr)
    return out, pieces, 2


def measure(name, base, mixed, pieces, spans, sr: int = SR) -> dict:
    track = cue_track(base.size, pieces, sr)
    clean = ig.assess(base, base)
    got = ig.assess(base, mixed)
    cues = [cue for _, cue in pieces]
    return {
        "name": name,
        "cue_count": len(cues),
        "speech_band": float(np.mean([accent.speech_band_fraction(c) for c in cues])),
        "overlap": accent.overlap_fraction(track, spans, sr),
        "sti": got.sti,
        "wer": got.wer,
        "clean_wer": clean.wer,
        "delta_wer": got.wer - clean.wer,
    }


def page(rows, out_dir: Path) -> str:
    body = []
    for row in rows:
        body.append(
            f"<section><h2>{html.escape(row['title'])}</h2>"
            f"<p class='note'>{html.escape(row['note'])}</p>"
            f"<audio controls preload='metadata' src='{row['file']}'></audio>"
            f"<table>"
            f"<tr><th>cues per turn</th><td>{row['cue_count']}</td></tr>"
            f"<tr><th>cue energy in 500&nbsp;Hz&ndash;2&nbsp;kHz</th>"
            f"<td>{row['speech_band'] * 100:.0f}%</td></tr>"
            f"<tr><th>cue energy while talking</th><td>{row['overlap'] * 100:.0f}%</td></tr>"
            f"<tr><th>est WER vs clean</th>"
            f"<td>{row['wer'] * 100:.1f}% vs {row['clean_wer'] * 100:.1f}%</td></tr>"
            f"</table></section>"
        )
    body.append(
        "<section><h2>clean reference</h2>"
        f"<audio controls preload='metadata' src='clean.wav'></audio></section>"
    )
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>vox accent options</title>"
        "<style>body{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;"
        "max-width:44rem;margin:2rem auto;padding:0 1rem;color:#111}"
        "h1{font-size:1.4rem}h2{font-size:1.05rem;margin:1.6rem 0 .2rem}"
        "section{border-top:1px solid #ddd;padding-top:.6rem}"
        ".note{color:#555;margin:.2rem 0 .6rem}"
        "audio{width:100%}table{border-collapse:collapse;margin-top:.5rem}"
        "th{text-align:left;font-weight:500;color:#555;padding-right:1rem}"
        "td{padding:.1rem 0}</style>"
        "<h1>vox accent options</h1>"
        "<p class='note'>Same voice throughout; the first three are a two-sentence "
        "answer, the last two a bare \u201cno\u201d. Cues sit at a fixed level below the "
        "voice and are measured with the preset guard.</p>"
        + "".join(body)
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/accent-demo")
    p.add_argument("--chime-db", type=float, default=-22.0)
    p.add_argument("--hum-db", type=float, default=-28.0)
    p.add_argument("--signature-db", type=float, default=-28.0)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    voice, voice_spans = build_voice()
    base = padded(voice)
    spans = [(a + LEAD, b + LEAD) for a, b in voice_spans]

    # the shortest thing a reply can be, so the one-word case is audible too
    short = testphrase.synthesize("no", sr=SR)
    short_base = padded(short)
    short_spans = [(LEAD, LEAD + short.size / SR)]

    configs = [
        ("chime at every sentence end",
         "A bell at each sentence boundary: it fires into the gap and its 0.9 s "
         "tail runs under the next sentence.",
         base, spans, lambda b, s: option_chime_per_sentence(b, s, args.chime_db),
         "opt1-chime-per-sentence.wav"),
        ("soft bass hum",
         "One continuous 55 Hz bed under the turn, at the level below. It is a "
         "presence bed, not a punctuation mark.",
         base, spans, lambda b, s: option_hum_bed(b, s, args.hum_db),
         "opt2-bass-hum.wav"),
        ("turn signature: breath in, settle out",
         "No chime. An inhale band before the first word and a low decay after "
         "the last one, one pair per turn regardless of how many sentences it has.",
         base, spans, lambda b, s: option_turn_signature(b, s, args.signature_db),
         "opt3-turn-signature.wav"),
        ("chime on a bare \u201cno\u201d",
         "The same bell on a one-word answer: one cue instead of one per sentence, "
         "and nothing after it for the tail to mask.",
         short_base, short_spans,
         lambda b, s: option_chime_per_sentence(b, s, args.chime_db),
         "opt1-chime-on-no.wav"),
        ("turn signature on a bare \u201cno\u201d",
         "Breath in, one word, settle out. Identical gesture, however short the answer.",
         short_base, short_spans,
         lambda b, s: option_turn_signature(b, s, args.signature_db),
         "opt3-turn-signature-on-no.wav"),
    ]

    rows = []
    for title, note, v_base, v_spans, build, filename in configs:
        mixed, pieces, count = build(v_base, v_spans)
        dsp.write_wav(out / filename, mixed, SR)
        row = measure(title, v_base, mixed, pieces, v_spans)
        row.update(title=title, note=note, file=filename, cue_count=count)
        rows.append(row)

    dsp.write_wav(out / "clean.wav", base, SR)
    (out / "index.html").write_text(page(rows, out), encoding="utf-8")

    print(f"{'option':34s} {'cues':>4s} {'in-band':>7s} {'overlap':>7s} "
          f"{'WER':>6s} {'delta':>6s}")
    for r in rows:
        print(f"{r['title']:34s} {r['cue_count']:4d} {r['speech_band'] * 100:6.0f}% "
              f"{r['overlap'] * 100:6.0f}% {r['wer'] * 100:5.1f}% "
              f"{r['delta_wer'] * 100:+5.1f}%")
    print(f"\nclean reference WER {rows[0]['clean_wer'] * 100:.1f}% "
          f"(two-sentence turn)")
    print(f"page: {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
