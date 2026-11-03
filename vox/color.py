"""The render chain: turn a source signal into a "colored" signal.

Chain order is fixed and documented:

    pitch shift (phase vocoder + resample)
      -> formant shift (cepstral envelope warp)
      -> EQ (STFT shelving + bell)
      -> drive -> ring mod -> tremolo -> comb
      -> bit crush -> noise bed -> output gain -> peak limit

Pitch is shifted by time-stretching at rate r and playing back at 1/r, which
moves formants with pitch (the physical "chipmunk" behaviour) and is then
optionally corrected by the independent formant warp.
"""

from __future__ import annotations

import numpy as np

from . import dsp
from .presets import Preset

SR = dsp.SR


# --------------------------------------------------------------------------
# pitch / formant
# --------------------------------------------------------------------------


def time_stretch(x, rate: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Phase-vocoder time stretch. ``rate`` = output_len / input_len."""
    x = np.asarray(x, dtype=float)
    if abs(rate - 1.0) < 1e-6:
        return x.copy()
    s = dsp.stft(x, n_fft, hop)
    n_frames, n_bins = s.spec.shape
    hs = int(round(hop * rate))
    hs = max(1, min(hs, n_fft // 2))
    omega = 2.0 * np.pi * np.arange(n_bins) / n_fft
    phase = np.angle(s.spec[0]).copy()
    prev = phase.copy()
    out = np.empty_like(s.spec)
    out[0] = s.spec[0]
    for i in range(1, n_frames):
        mag = np.abs(s.spec[i])
        ph = np.angle(s.spec[i])
        delta = ph - prev - omega * hop
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        phase = phase + (omega + delta / hop) * hs
        prev = ph
        out[i] = mag * np.exp(1j * phase)
    target = int(round(x.size * rate))
    y = dsp.istft(dsp.STFT(out, n_fft, hs, 0), None)
    # frames were analysed from a signal padded by n_fft, so the leading
    # n_fft samples are the pad: trim them to keep the stretch time-aligned
    if y.size > n_fft:
        y = y[n_fft:]
    if y.size < target:
        y = np.pad(y, (0, target - y.size))
    return y[:target]


def shift_pitch(x, semitones: float, sr: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if abs(semitones) < 0.02:
        return x.copy()
    r = 2.0 ** (float(semitones) / 12.0)
    stretched = time_stretch(x, r)  # same pitch, r times longer
    return dsp.resample_ratio(stretched, 1.0 / r, sr)  # pitch * r, original length


def shift_formants(x, ratio: float, sr: int = SR) -> np.ndarray:
    """Warp the spectral envelope by ``ratio`` while keeping pitch and length."""
    x = np.asarray(x, dtype=float)
    if abs(ratio - 1.0) < 2e-3:
        return x.copy()
    n_fft, hop = 1024, 256
    s = dsp.stft(x, n_fft, hop)
    log_mag = np.log(np.abs(s.spec) + 1e-7)
    cep = np.fft.irfft(log_mag, n=n_fft, axis=1)
    lifter = int(2.5e-3 * sr)  # 2.5 ms -> spectral envelope
    lifter = max(4, min(lifter, n_fft // 4))
    cep[:, lifter : n_fft - lifter] = 0.0
    env = np.exp(np.fft.rfft(cep, axis=1).real)
    freqs = s.freqs
    warped = np.empty_like(env)
    for i in range(env.shape[0]):
        warped[i] = np.interp(freqs / ratio, freqs, env[i])
    mag = np.abs(s.spec) / np.maximum(env, 1e-9) * np.maximum(warped, 1e-9)
    return dsp.istft(
        dsp.STFT(mag * np.exp(1j * np.angle(s.spec)), n_fft, hop, s.offset), x.size
    )


# --------------------------------------------------------------------------
# EQ
# --------------------------------------------------------------------------


def eq_gain(freqs, params: dict) -> np.ndarray:
    """Magnitude response (linear) of the preset's three-band EQ."""
    f = np.maximum(np.asarray(freqs, dtype=float), 1.0)
    low = 1.0 / (1.0 + (f / 250.0) ** 2)
    high = 1.0 / (1.0 + (4200.0 / f) ** 2)
    width = 1.0 / max(params["peak_q"], 0.1)
    bell = np.exp(-0.5 * (np.log2(f / params["peak_freq"]) / (0.7 * width)) ** 2)
    db = params["low_shelf_db"] * low + params["high_shelf_db"] * high + params["peak_db"] * bell
    return dsp.db_to_gain(db)


# --------------------------------------------------------------------------
# chain
# --------------------------------------------------------------------------


def render(x, preset, sr: int = SR, rng=None) -> np.ndarray:
    """Apply a preset (a ``Preset`` or a params dict) to a mono signal."""
    params = preset.params if isinstance(preset, Preset) else dict(preset)
    rng = np.random.default_rng(0) if rng is None else rng
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    if y.size == 0:
        return y

    if abs(params["pitch_semitones"]) > 0.02:
        y = shift_pitch(y, params["pitch_semitones"], sr)
    if abs(params["formant_shift"] - 1.0) > 2e-3:
        y = shift_formants(y, params["formant_shift"], sr)

    y = dsp.apply_spectral_gain(y, lambda f, i: eq_gain(f, params), 1024, 256)

    if params["drive"] > 0.005:
        y = dsp.drive(y, params["drive"])
    if params["ring_amount"] > 0.005:
        y = dsp.ring_mod(y, params["ring_hz"], params["ring_amount"], sr)
    if params["tremolo_depth"] > 0.005 and params["tremolo_hz"] > 0.05:
        y = dsp.tremolo(y, params["tremolo_hz"], params["tremolo_depth"], sr)
    if params["comb_mix"] > 0.005:
        y = dsp.comb(y, params["comb_delay_ms"], params["comb_mix"], 0.3, sr)
    if params["crush_bits"] < 15.5:
        y = dsp.crush(y, int(round(params["crush_bits"])))
    if params["noise_mix"] > 0.002:
        y = y + dsp.noise_bed(y.size, params["noise_mix"], sr, rng)

    y = y * float(dsp.db_to_gain(params["output_gain_db"]))
    return dsp.peak_limit(y)


def render_file(src_path, preset, out_path, sr: int = SR) -> "dsp.Path":
    x, rate = dsp.read_wav(src_path, sr)
    y = render(x, preset, rate)
    return dsp.write_wav(out_path, y, rate)
