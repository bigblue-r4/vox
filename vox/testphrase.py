"""A self-contained formant synthesizer for the standard test phrase.

No recording and no external audio: the phrase is spoken by a small source
(filter) model driven by a phoneme table. It is deliberately a fixed, identical
carrier for every preset, so A/B auditions differ only by the color.

Pipeline: build per-frame voicing / frication / formant tracks from the
phoneme sequence, render an excitation (glottal pulse train + gated noise),
then shape it with the formant envelope in the STFT domain. ``--speech`` lets
you substitute a real recording for the guard and the auditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import TEST_PHRASE, dsp

SR = dsp.SR
FRAME_S = 0.005  # 5 ms control rate, matches the 128-hop STFT analysis


@dataclass
class Phone:
    kind: str  # "vowel" | "fricative" | "stop" | "nasal" | "liquid" | "glide"
    formants: tuple[float, float, float]
    dur: float
    voiced: float = 1.0
    fric: tuple[float, float, float] | None = None  # (lo_hz, hi_hz, gain)
    glide_to: tuple[float, float, float] | None = None


PHONES: dict[str, Phone] = {
    # vowels
    "AA": Phone("vowel", (730, 1090, 2440), 0.115),
    "AE": Phone("vowel", (660, 1720, 2410), 0.110),
    "AH": Phone("vowel", (640, 1190, 2390), 0.080),
    "AO": Phone("vowel", (570, 840, 2410), 0.110),
    "AW": Phone("vowel", (700, 1100, 2400), 0.135, glide_to=(400, 800, 2200)),
    "EH": Phone("vowel", (530, 1840, 2480), 0.100),
    "ER": Phone("vowel", (490, 1350, 1690), 0.110),
    "EY": Phone("vowel", (530, 1840, 2480), 0.135, glide_to=(300, 2200, 2900)),
    "IH": Phone("vowel", (390, 1990, 2550), 0.075),
    "IY": Phone("vowel", (270, 2290, 3010), 0.105),
    "OW": Phone("vowel", (570, 840, 2410), 0.130, glide_to=(330, 900, 2250)),
    "UH": Phone("vowel", (440, 1020, 2240), 0.075),
    "UW": Phone("vowel", (300, 870, 2240), 0.100),
    # nasals / liquids / glides
    "M": Phone("nasal", (250, 1100, 2200), 0.070, voiced=0.75),
    "N": Phone("nasal", (250, 1400, 2400), 0.065, voiced=0.75),
    "NG": Phone("nasal", (250, 1600, 2500), 0.070, voiced=0.75),
    "L": Phone("liquid", (360, 1300, 2700), 0.060, voiced=0.9),
    "R": Phone("liquid", (310, 1060, 1380), 0.065, voiced=0.9),
    "W": Phone("glide", (300, 610, 2200), 0.055, voiced=0.9),
    "Y": Phone("glide", (280, 2200, 3000), 0.055, voiced=0.9),
    # fricatives
    "S": Phone("fricative", (320, 1600, 2600), 0.095, voiced=0.0, fric=(4200, 9000, 0.85)),
    "Z": Phone("fricative", (300, 1500, 2500), 0.080, voiced=0.35, fric=(3600, 7800, 0.55)),
    "F": Phone("fricative", (300, 1100, 2400), 0.085, voiced=0.10, fric=(900, 6000, 0.45)),
    "V": Phone("fricative", (300, 1150, 2400), 0.070, voiced=0.45, fric=(800, 4200, 0.35)),
    "TH": Phone("fricative", (300, 1400, 2600), 0.080, voiced=0.0, fric=(3000, 8000, 0.5)),
    "DH": Phone("fricative", (300, 1400, 2500), 0.055, voiced=0.5, fric=(1200, 5000, 0.3)),
    "SH": Phone("fricative", (300, 1800, 2600), 0.095, voiced=0.0, fric=(2200, 6500, 0.8)),
    "HH": Phone("fricative", (500, 1500, 2500), 0.065, voiced=0.0, fric=(500, 3500, 0.4)),
    # stops (closure + burst)
    "P": Phone("stop", (300, 900, 2200), 0.080, voiced=0.0, fric=(400, 2500, 0.6)),
    "B": Phone("stop", (250, 900, 2200), 0.070, voiced=0.4, fric=(250, 1200, 0.45)),
    "T": Phone("stop", (350, 1700, 2600), 0.075, voiced=0.0, fric=(1500, 5000, 0.6)),
    "D": Phone("stop", (300, 1600, 2500), 0.065, voiced=0.4, fric=(900, 3500, 0.45)),
    "K": Phone("stop", (350, 1800, 2600), 0.080, voiced=0.0, fric=(1200, 4000, 0.6)),
    "G": Phone("stop", (300, 1700, 2500), 0.070, voiced=0.4, fric=(800, 3000, 0.45)),
    "CH": Phone("stop", (300, 1800, 2600), 0.095, voiced=0.0, fric=(2200, 6500, 0.8)),
    "JH": Phone("stop", (300, 1700, 2500), 0.090, voiced=0.4, fric=(2000, 6000, 0.6)),
}

WORDS: dict[str, list[str]] = {
    "the": ["DH", "AH"],
    "quick": ["K", "W", "IH", "K"],
    "brown": ["B", "R", "AW", "N"],
    "fox": ["F", "AA", "K", "S"],
    "jumps": ["JH", "AH", "M", "P", "S"],
    "over": ["OW", "V", "ER"],
    "lazy": ["L", "EY", "Z", "IY"],
    "dog": ["D", "AO", "G"],
    # a one-word answer, so turn-edge cues can be measured on the shortest
    # thing a reply can be
    "no": ["N", "OW"],
}

WORD_GAP_S = 0.045
LEAD_S = 0.10
TAIL_S = 0.16


@dataclass
class _Track:
    """Per-frame control values, one entry per 5 ms frame."""

    f1: list[float] = field(default_factory=list)
    f2: list[float] = field(default_factory=list)
    f3: list[float] = field(default_factory=list)
    voiced: list[float] = field(default_factory=list)
    fric_lo: list[float] = field(default_factory=list)
    fric_hi: list[float] = field(default_factory=list)
    fric_gain: list[float] = field(default_factory=list)
    amp: list[float] = field(default_factory=list)

    def n(self) -> int:
        return len(self.f1)


def _phone_frames(phone: Phone, index: int, total: int) -> list[dict]:
    frames = []
    n = max(1, int(round(phone.dur / FRAME_S)))
    for i in range(n):
        u = i / max(1, n - 1)
        if phone.glide_to is not None:
            target = phone.glide_to
            # diphthongs: hold the onset, then glide over the second half
            k = max(0.0, (u - 0.45) / 0.55)
            fmt = tuple(
                a + (b - a) * k for a, b in zip(phone.formants, target)
            )
        else:
            fmt = phone.formants
        voiced = phone.voiced
        fric_gain = 0.0
        fric_lo, fric_hi = 2000.0, 6000.0
        amp = 1.0
        if phone.kind == "stop":
            # closure then burst: silence for the first 60%, noise for the rest
            fric_gain = phone.fric[2] if phone.fric and u > 0.60 else 0.0
            if phone.fric:
                fric_lo, fric_hi = phone.fric[0], phone.fric[1]
            voiced = 0.0
            amp = 0.85 if u > 0.60 else 0.05
        elif phone.kind == "fricative" and phone.fric:
            fric_gain = phone.fric[2]
            fric_lo, fric_hi = phone.fric[0], phone.fric[1]
            amp = 0.8
        elif phone.kind in {"nasal", "liquid", "glide"}:
            amp = 0.62
        frames.append(
            {
                "f": fmt,
                "voiced": voiced,
                "fric": (fric_lo, fric_hi, fric_gain),
                "amp": amp,
            }
        )
    return frames


def _smooth(values: list[float], frames: int = 3) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size < frames or frames < 2:
        return v
    kernel = np.ones(frames) / frames
    padded = np.pad(v, (frames // 2, frames // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: v.size]


def build_tracks(text: str = TEST_PHRASE) -> _Track:
    track = _Track()
    frames: list[dict] = []
    frames.extend([{"sil": True}] * int(round(LEAD_S / FRAME_S)))
    words = text.lower().split()
    for w_index, word in enumerate(words):
        phones = WORDS.get(word)
        if phones is None:
            raise ValueError(f"no phoneme entry for word {word!r}")
        for i, name in enumerate(phones):
            if name not in PHONES:
                raise ValueError(f"unknown phone {name!r}")
            frames.extend(_phone_frames(PHONES[name], i, len(phones)))
        if w_index < len(words) - 1:
            frames.extend([{"sil": True}] * int(round(WORD_GAP_S / FRAME_S)))
    frames.extend([{"sil": True}] * int(round(TAIL_S / FRAME_S)))

    for fr in frames:
        if "sil" in fr:
            track.f1.append(500.0)
            track.f2.append(1500.0)
            track.f3.append(2500.0)
            track.voiced.append(0.0)
            track.fric_gain.append(0.0)
            track.fric_lo.append(2000.0)
            track.fric_hi.append(6000.0)
            track.amp.append(0.0)
        else:
            track.f1.append(fr["f"][0])
            track.f2.append(fr["f"][1])
            track.f3.append(fr["f"][2])
            track.voiced.append(fr["voiced"])
            track.fric_lo.append(fr["fric"][0])
            track.fric_hi.append(fr["fric"][1])
            track.fric_gain.append(fr["fric"][2])
            track.amp.append(fr["amp"])
    return track


def synthesize(text: str = TEST_PHRASE, f0: float = 118.0, sr: int = SR,
               n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Render the test phrase as mono float64 at ``sr``."""
    track = build_tracks(text)
    n_frames = track.n()
    n = int(round(n_frames * FRAME_S * sr))
    t = np.arange(n) / sr

    # --- per-sample control tracks -------------------------------------
    frame_times = (np.arange(n_frames) + 0.5) * FRAME_S
    amp = np.interp(t, frame_times, _smooth(track.amp, 3))
    voiced = np.interp(t, frame_times, _smooth(track.voiced, 3))
    fric_gain = np.interp(t, frame_times, _smooth(track.fric_gain, 3))
    fric_lo = np.interp(t, frame_times, _smooth(track.fric_lo, 5))
    fric_hi = np.interp(t, frame_times, _smooth(track.fric_hi, 5))

    # --- prosody: declination, final fall, gentle micro-variation ------
    total = n / sr
    decl = f0 * (1.0 - 0.18 * (t / max(total, 1e-6)))
    fall = 1.0 - 0.10 * np.clip((t - (total - 0.45)) / 0.45, 0.0, 1.0)
    micro = 1.0 + 0.008 * np.sin(2.0 * np.pi * 4.3 * t) + 0.004 * np.sin(
        2.0 * np.pi * 0.7 * t
    )
    f0_track = decl * fall * micro

    # --- excitation: glottal pulses + gated noise ---------------------
    exc = np.zeros(n)
    phase = 0.0
    i = 0
    while i < n:
        exc[i] += float(voiced[i]) * float(amp[i])
        phase += f0_track[i] / sr
        step = max(1, int(round(sr / max(f0_track[i], 60.0))))
        i += step

    # glottal tilt: pulses are flat-spectrum, roll them off like a glottal source
    exc = dsp.lowpass_fft(exc, 3200.0, sr)

    rng = np.random.default_rng(7)  # fixed: the phrase is identical every run
    noise = rng.standard_normal(n)

    # --- formant / frication shaping in the STFT domain ---------------
    s = dsp.stft(exc, n_fft, hop)
    sf = dsp.stft(noise, n_fft, hop)
    freqs = s.freqs
    out_frames = np.empty_like(s.spec)
    n_stft = s.spec.shape[0]
    for k in range(n_stft):
        # control frame whose centre is nearest this STFT frame
        j = int(np.clip(round((k * hop - s.offset) / (FRAME_S * sr)), 0, n_frames - 1))
        # formants with fixed bandwidths -> resonance response
        gain = (
            1.0 * _resonance(freqs, track.f1[j], 90.0)
            + 0.55 * _resonance(freqs, track.f2[j], 110.0)
            + 0.30 * _resonance(freqs, track.f3[j], 170.0)
        )
        gain = gain / (1.0 + (freqs / 5200.0) ** 2)
        a = float(track.amp[j])
        out_frames[k] = s.spec[k] * gain * a
        # frication: band noise gated by this frame's fricative settings
        g = float(track.fric_gain[j])
        if g > 0.001 and a > 0.0:
            band = _band_window(freqs, float(track.fric_lo[j]), float(track.fric_hi[j]))
            out_frames[k] += sf.spec[k] * band * g * 0.55

    y = dsp.istft(dsp.STFT(out_frames, n_fft, hop, s.offset), n)
    y = dsp.fade(np.tanh(y * 1.2), 8.0, 30.0)
    return dsp.normalize(y, dsp.PEAK)


def _resonance(freqs: np.ndarray, center: float, bandwidth: float) -> np.ndarray:
    f = np.maximum(freqs, 1.0)
    x = f / max(center, 1.0)
    denom = np.sqrt((1.0 - x**2) ** 2 + (f / (max(center, 1.0) * (bandwidth / 100.0))) ** 2)
    return 1.0 / np.maximum(denom, 1e-6)


def _band_window(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f = np.maximum(freqs, 1.0)
    with np.errstate(divide="ignore"):
        a = 1.0 / (1.0 + np.exp(np.clip(np.log2(f / hi) * 10.0, -60, 60)))
        b = 1.0 / (1.0 + np.exp(np.clip(np.log2(lo / f) * 10.0, -60, 60)))
    return a * b


def write_phrase(path, sr: int = SR) -> "object":
    y = synthesize(TEST_PHRASE, sr=sr)
    return dsp.write_wav(path, y, sr)
