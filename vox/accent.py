"""Non-speech accents: sound cues that mark a turn without masking the voice.

Three families, all synthesized here so an accent is a reproducible artifact
rather than a sample library:

``chime``
    An inharmonic bell (partials at 1 : 2.76 : 5.40 : 8.93, like a struck
    bar). Meant to be one per *turn*, not per sentence: see ``analyse``.

``hum``
    A low bed: a sine fundamental with a weak second harmonic and a slow
    envelope. Placed under the voice, not mixed into it.

``breath`` / ``settle``
    The turn-edge pair. ``breath`` is a short inhale-shaped noise burst in
    2.5-7.5 kHz placed just before the first word; ``settle`` is a low
    (60-90 Hz) decaying body placed just after the last one. Neither occupies
    the 500 Hz - 2 kHz region that carries most speech cues, and ``settle`` has
    no pitched attack, which is what keeps it from reading as a notification.

The accent is *mixed under* speech at a level relative to the voice's RMS, so
"soft" is a measurable quantity rather than an adjective: ``level_db=-26`` means
the cue sits 26 dB below the voice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dsp

SR = dsp.SR

#: Partials of a struck bar, relative to the fundamental: (ratio, gain).
BELL_PARTIALS = ((1.0, 1.0), (2.76, 0.42), (5.40, 0.18), (8.93, 0.07))


def _raised_cosine(n: int) -> np.ndarray:
    """Half-cosine rise, so a cue never starts with a click."""
    if n <= 0:
        return np.zeros(0)
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)


def _envelope(n: int, attack: float, release: float, sr: int = SR) -> np.ndarray:
    env = np.ones(n)
    a = min(n, max(1, int(attack * sr)))
    r = min(n, max(1, int(release * sr)))
    env[:a] *= _raised_cosine(a)
    env[n - r :] *= _raised_cosine(r)[::-1]
    return env


def chime(
    dur: float = 0.9,
    hz: float = 1046.5,
    sr: int = SR,
    decay: float = 0.30,
    partials=BELL_PARTIALS,
) -> np.ndarray:
    """An inharmonic bell at ``hz``.

    Higher partials decay faster, which is what makes it read as struck metal
    instead of a chord: the shimmer arrives first and the fundamental rings on.
    """
    n = max(16, int(dur * sr))
    t = np.arange(n) / sr
    y = np.zeros(n)
    for ratio, gain in partials:
        f = hz * ratio
        if f >= 0.45 * sr:
            continue  # would alias
        tau = decay / (1.0 + 0.7 * (ratio - 1.0))
        y += gain * np.sin(2.0 * np.pi * f * t) * np.exp(-t / max(tau, 1e-3))
    y *= _envelope(n, 0.003, 0.02, sr)
    return dsp.normalize(y, 0.891)


def hum(
    dur: float = 1.4,
    hz: float = 55.0,
    sr: int = SR,
    harmonic: float = 0.30,
) -> np.ndarray:
    """A soft low bed: sine plus a weak second harmonic.

    Kept under ~90 Hz on purpose. Above roughly 120 Hz a steady tone lands in
    the neighbourhood of the talker's own fundamental, where it stops being a
    bed and starts being a second, competing pitch.
    """
    n = max(16, int(dur * sr))
    t = np.arange(n) / sr
    y = np.sin(2.0 * np.pi * hz * t)
    y += harmonic * np.sin(2.0 * np.pi * 2.0 * hz * t + 0.4)
    y *= _envelope(n, 0.18, 0.30, sr)
    return dsp.normalize(y, 0.891)


def breath(dur: float = 0.24, sr: int = SR, rng=None, lo_hz: float = 2500.0,
           hi_hz: float = 7500.0) -> np.ndarray:
    """An inhale-shaped noise burst, band-limited to the air region.

    This is the 'about to speak' tell. It sits above the formant range, so it
    cannot mask a vowel even when it is loud enough to notice.
    """
    rng = np.random.default_rng(7) if rng is None else rng
    n = max(16, int(dur * sr))
    nz = rng.standard_normal(n)
    nz = nz - dsp.lowpass_fft(nz, lo_hz, sr)
    nz = dsp.lowpass_fft(nz, hi_hz, sr)
    t = np.linspace(0.0, 1.0, n)
    # breath moves fastest at the start and eases out: an inhale, not a hit
    shape = np.sin(np.pi * np.clip(t ** 0.75, 0.0, 1.0)) ** 2
    y = nz * shape
    return dsp.normalize(y, 0.891)


def settle(dur: float = 0.60, hz: float = 74.0, sr: int = SR) -> np.ndarray:
    """A low, unpitched-feeling decay that closes the turn.

    Two partials an octave apart with a fast exponential decay read as a body
    settling rather than a note being played, and the 20 ms attack removes the
    transient that would make it a knock.
    """
    n = max(16, int(dur * sr))
    t = np.arange(n) / sr
    y = np.sin(2.0 * np.pi * hz * t) * np.exp(-t / 0.11)
    y += 0.35 * np.sin(2.0 * np.pi * 2.0 * hz * t + 0.2) * np.exp(-t / 0.06)
    y *= _envelope(n, 0.020, 0.03, sr)
    return dsp.normalize(y, 0.891)


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


@dataclass
class Placement:
    """Where a cue sits relative to the voice, in seconds."""

    at: float
    level_db: float = -26.0


def place(voice, cue, at_s: float, level_db: float = -26.0, sr: int = SR) -> np.ndarray:
    """Mix ``cue`` into ``voice`` starting ``at_s`` seconds in.

    The cue is scaled relative to the *voice's* RMS, so ``level_db`` is a level
    difference and not an absolute one: the accent stays the same distance below
    the speech no matter how loud the voice is rendered.
    """
    voice = np.asarray(voice, dtype=float)
    cue = np.asarray(cue, dtype=float)
    if at_s < 0:
        raise ValueError("cue offset cannot be negative")
    start = int(round(at_s * sr))
    n = max(voice.size, start + cue.size)
    out = np.zeros(n)
    out[: voice.size] = voice
    v_rms = dsp.rms(voice)
    c_rms = dsp.rms(cue)
    if c_rms > 1e-12 and v_rms > 1e-12:
        cue = cue * (v_rms * float(dsp.db_to_gain(level_db)) / c_rms)
    out[start : start + cue.size] += cue
    return out


def turn_signature(
    voice,
    sr: int = SR,
    lead: float = 0.30,
    tail: float = 0.55,
    level_db: float = -30.0,
    rng=None,
) -> np.ndarray:
    """Wrap a spoken turn in a breath and a settle.

    Returns audio that starts ``lead`` seconds before the first word and ends
    ``tail`` seconds after the last, so the cue has room to happen instead of
    being trimmed off by whatever plays the file next.
    """
    voice = np.asarray(voice, dtype=float)
    inhale = breath(0.24, sr, rng)
    # the inhale has to *finish* before the first word: breath is anticipation,
    # and anticipation landing on top of the first syllable is just noise
    out = np.concatenate([np.zeros(int(round(lead * sr))), voice,
                          np.zeros(int(round(tail * sr)))])
    out = place(out, inhale, at_s=0.0, level_db=level_db, sr=sr)
    out = place(out, settle(0.60, sr=sr), at_s=lead + voice.size / sr + 0.02,
                level_db=level_db - 2.0, sr=sr)
    return out


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def band_energy(x, lo_hz: float, hi_hz: float, sr: int = SR) -> float:
    """RMS of ``x`` inside a band, using the same soft FFT skirts as the guard."""
    x = np.asarray(x, dtype=float)
    if x.size < 32:
        return dsp.rms(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    with np.errstate(divide="ignore"):
        above = 1.0 / (1.0 + np.exp(np.clip(np.log2(np.maximum(f, 1e-9) / hi_hz) * 8.0, -60, 60)))
        below = 1.0 / (1.0 + np.exp(np.clip(np.log2(lo_hz / np.maximum(f, 1e-9)) * 8.0, -60, 60)))
    return dsp.rms(np.fft.irfft(X * above * below, x.size))


def speech_band_fraction(cue, sr: int = SR) -> float:
    """Fraction of a cue's energy inside 500 Hz - 2 kHz, where speech lives."""
    total = dsp.rms(cue) ** 2
    if total < 1e-18:
        return 0.0
    return float(min(1.0, band_energy(cue, 500.0, 2000.0, sr) ** 2 / total))


def overlap_fraction(track, voice_spans, sr: int = SR) -> float:
    """Fraction of a cue track's energy that lands while the talker is talking.

    ``voice_spans`` is a list of ``(start_s, end_s)``. A cue that fires at a
    sentence boundary spends its tail under the *next* sentence, and this is how
    much of it does.
    """
    track = np.asarray(track, dtype=float)
    mask = np.zeros(track.size, dtype=bool)
    for start_s, end_s in voice_spans:
        a = max(0, int(round(start_s * sr)))
        b = min(track.size, int(round(end_s * sr)))
        if b > a:
            mask[a:b] = True
    total = float(np.sum(track**2))
    if total < 1e-18:
        return 0.0
    return float(np.sum(track[mask] ** 2) / total)
