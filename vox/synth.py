"""Procedural sample synthesis — six original "color sample" sources.

Every source is generated from code (no external audio, no samples on disk):
FM bleeps/chirps, filtered noise sweeps, metallic modal resonators, granular
clouds built from synthesized tones, pulse/click trains, and glassy additive
pads. Each function takes a numpy Generator and returns mono float64 at
``dsp.SR``, peak-normalised with short fades.
"""

from __future__ import annotations

import numpy as np

from . import dsp

SR = dsp.SR
PEAK = dsp.PEAK * 0.98

SOURCES = (
    "fm-bleep",
    "noise-sweep",
    "modal-bell",
    "granular-cloud",
    "pulse-train",
    "additive-pad",
)

#: What each source sounds like, used by the shortlist and the README.
SOURCE_BLURB = {
    "fm-bleep": "FM bleeps and chirps with decaying modulation index",
    "noise-sweep": "band-passed noise sweeping through the spectrum",
    "modal-bell": "struck-bar / bell modes, inharmonic partials",
    "granular-cloud": "grain cloud of synthesized tones in a pentatonic field",
    "pulse-train": "click and pulse trains through a resonant band",
    "additive-pad": "glassy additive pad, detuned partial stack",
}


def _t(dur: float) -> np.ndarray:
    return np.arange(int(round(dur * SR))) / SR


def _finish(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    return dsp.normalize(dsp.fade(x, 4.0, 25.0), PEAK)


def fm_bleep(rng: np.random.Generator, dur: float = 2.4) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    pos = 0.0
    while pos < dur - 0.05:
        length = float(rng.uniform(0.10, 0.40))
        carrier = float(rng.uniform(180.0, 2000.0))
        ratio = float(rng.choice([1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 7.0]))
        index = float(rng.uniform(1.0, 7.0))
        glide = float(rng.uniform(-0.6, 0.9))  # octaves across the bleep
        start = int(pos * SR)
        seg_n = min(int(length * SR), n - start)
        if seg_n <= 16:
            break
        ts = np.arange(seg_n) / SR
        f0 = carrier * 2.0 ** (glide * ts / max(length, 1e-6))
        phase = 2.0 * np.pi * np.cumsum(f0) / SR
        mod_index = index * np.exp(-ts * 2.5 / length)
        seg = np.sin(phase + mod_index * np.sin(2.0 * np.pi * np.cumsum(f0 * ratio) / SR))
        env = np.minimum(1.0, ts / 0.004) * np.exp(-ts * 3.0 / length)
        out[start : start + seg_n] += seg * env * float(rng.uniform(0.4, 1.0))
        pos += length + float(rng.uniform(0.02, 0.20))
    return _finish(out)


def noise_sweep(rng: np.random.Generator, dur: float = 2.5) -> np.ndarray:
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    f_start = float(rng.uniform(180.0, 500.0))
    f_end = float(rng.uniform(2500.0, 9500.0))
    if rng.random() < 0.45:
        f_start, f_end = f_end, f_start
    bw = float(rng.uniform(0.35, 1.15))  # octaves, half-width
    shape = float(rng.uniform(0.6, 1.0))

    s = dsp.stft(noise, 1024, 256)
    n_frames = s.spec.shape[0]
    prog = np.linspace(0.0, 1.0, n_frames)
    centers = np.log2(f_start) + prog * np.log2(f_end / f_start)
    logf = np.log2(np.maximum(s.freqs, 1e-9))
    band = np.exp(-0.5 * ((logf[None, :] - centers[:, None]) / bw) ** 2)
    amp = np.sin(np.pi * prog) ** shape
    out = dsp.istft(
        dsp.STFT(s.spec * band * amp[:, None], s.n_fft, s.hop, s.offset), n
    )
    # a thin tonal shadow of the sweep keeps it from sounding like pure hiss
    if rng.random() < 0.7:
        t = np.arange(n) / SR
        f_track = f_start * (f_end / f_start) ** (t / dur)
        tone = 0.18 * np.sin(2.0 * np.pi * np.cumsum(f_track) / SR)
        out = out + tone * np.sin(np.pi * np.minimum(1.0, t / dur))
    return _finish(out)


def modal_bell(rng: np.random.Generator, dur: float = 2.6) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    base = float(rng.uniform(120.0, 880.0))
    ratios = np.array(
        [
            1.0,
            rng.uniform(2.2, 2.9),
            rng.uniform(4.7, 5.7),
            rng.uniform(8.0, 9.4),
            rng.uniform(12.0, 14.5),
        ]
    )
    amps = np.array([1.0, 0.62, 0.45, 0.28, 0.15]) * rng.uniform(0.7, 1.3, 5)
    taus = np.array([1.5, 1.05, 0.72, 0.46, 0.30]) * rng.uniform(0.6, 1.5, 5)
    out = np.zeros(n)
    for ratio, amp, tau in zip(ratios, amps, taus):
        f = base * ratio * float(rng.uniform(0.995, 1.005))
        if f > 0.45 * SR:
            continue
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        out += amp * np.sin(2.0 * np.pi * f * t + phase) * np.exp(-t / max(tau, 0.05))

    # strike excitation
    click_n = int(0.007 * SR)
    click = rng.standard_normal(click_n) * np.exp(-np.arange(click_n) / (0.0016 * SR))
    out[:click_n] += 2.2 * dsp.lowpass_fft(click, 6500.0, SR)

    # slow beating between two nearby modes (the "shimmer")
    beat_hz = float(rng.uniform(0.5, 3.5))
    out *= 1.0 + 0.12 * np.sin(2.0 * np.pi * beat_hz * t)
    return _finish(out)


def granular_cloud(rng: np.random.Generator, dur: float = 2.6) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    scale = 2.0 ** (np.array([0, 2, 4, 7, 9, 12, 14, 16, 19, 24]) / 12.0)
    root = float(rng.uniform(110.0, 440.0))
    density = float(rng.uniform(18.0, 55.0))
    base_grain = int(0.06 * SR)
    for _ in range(int(density * dur)):
        length = (
            base_grain
            if rng.random() < 0.7
            else int(base_grain * float(rng.uniform(0.4, 1.6)))
        )
        length = max(8, length)
        start = int(rng.uniform(0, max(1, n - length)))
        ts = np.arange(length) / SR
        f = root * float(rng.choice(scale)) * 2.0 ** int(rng.integers(-1, 2))
        if f > 0.45 * SR:
            continue
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)
        env = window * np.exp(-ts / float(rng.uniform(0.02, 0.09)))
        grain = np.sin(2.0 * np.pi * f * ts + float(rng.uniform(0, 2 * np.pi)))
        grain += 0.3 * np.sin(2.0 * np.pi * 2.0 * f * ts)
        out[start : start + length] += grain * env * float(rng.uniform(0.35, 1.0))
    return _finish(np.tanh(out * 0.8))


def pulse_train(rng: np.random.Generator, dur: float = 2.2) -> np.ndarray:
    n = int(dur * SR)
    rate = float(rng.uniform(4.0, 30.0))
    period = max(2, int(SR / rate))
    exc = np.zeros(n)
    pos = 0
    while pos < n:
        exc[pos] = 1.0
        pos += max(2, int(period * float(rng.uniform(0.85, 1.15))))
    click_n = int(0.004 * SR)
    exc = np.convolve(exc, np.exp(-np.arange(click_n) / (0.0012 * SR)))[:n]

    center = float(rng.uniform(300.0, 3000.0))
    bw = float(rng.uniform(0.4, 1.0))
    logf = np.log2(np.maximum(np.fft.rfftfreq(1024, 1.0 / SR), 1e-9))
    band = np.exp(-0.5 * ((logf - np.log2(center)) / bw) ** 2)
    lift = 10.0 ** (float(rng.uniform(6.0, 14.0)) / 20.0)
    out = dsp.apply_spectral_gain(exc, lambda f, i: band * lift, 1024, 256)
    return _finish(out)


def additive_pad(rng: np.random.Generator, dur: float = 2.8) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    root = float(rng.uniform(70.0, 320.0))
    n_partials = int(rng.integers(6, 22))
    out = np.zeros(n)
    for i in range(1, n_partials + 1):
        f = root * i * float(rng.uniform(0.997, 1.008))
        if f > 0.45 * SR:
            break
        amp = 1.0 / (i ** float(rng.uniform(0.8, 1.6)))
        out += amp * np.sin(2.0 * np.pi * f * t + float(rng.uniform(0, 2 * np.pi)))
        if rng.random() < 0.5:  # detuned twin -> slow chorus beating
            f2 = f * float(rng.uniform(1.001, 1.006))
            out += 0.5 * amp * np.sin(2.0 * np.pi * f2 * t)
    env = np.minimum(1.0, t / float(rng.uniform(0.4, 1.2))) * np.minimum(
        1.0, np.maximum(0.0, (dur - t) / float(rng.uniform(0.3, 0.9)))
    )
    out *= env * (1.0 + 0.14 * np.sin(2.0 * np.pi * float(rng.uniform(0.2, 1.4)) * t))
    out = dsp.lowpass_fft(out, float(rng.uniform(3000.0, 9000.0)), SR)
    return _finish(out)


_GENERATORS = {
    "fm-bleep": fm_bleep,
    "noise-sweep": noise_sweep,
    "modal-bell": modal_bell,
    "granular-cloud": granular_cloud,
    "pulse-train": pulse_train,
    "additive-pad": additive_pad,
}


def synthesize(source: str, rng: np.random.Generator, dur: float | None = None) -> np.ndarray:
    """Render one sample for ``source``; deterministic for a given rng state."""
    if source not in _GENERATORS:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    dur = float(rng.uniform(2.0, 2.8)) if dur is None else float(dur)
    return _GENERATORS[source](rng, dur)


def source_for_index(i: int) -> str:
    """Balanced coverage: cycle through all six sources."""
    return SOURCES[i % len(SOURCES)]
