"""Low-level DSP primitives: WAV IO, STFT/ISTFT, resampling, time-domain effects.

Everything is mono float64 at 24 kHz unless stated otherwise. No scipy: the
FFT, the overlap-add machinery and the filters here are numpy-only so the
project installs with a single dependency.
"""

from __future__ import annotations

import hashlib
import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 24000
PEAK = 0.891  # -1 dBFS, headroom used for every rendered file


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------


def db_to_gain(db) -> np.ndarray:
    return 10.0 ** (np.asarray(db, dtype=float) / 20.0)


def gain_to_db(gain) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(gain, dtype=float)), 1e-12))


def rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def peak(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x))) if x.size else 0.0


def normalize(x, target_peak: float = PEAK) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = peak(x)
    return x if p < 1e-12 else x * (target_peak / p)


def peak_limit(x, ceiling: float = 0.99) -> np.ndarray:
    """Scale down only when the signal would clip."""
    x = np.asarray(x, dtype=float)
    p = peak(x)
    return x if p <= ceiling or p < 1e-12 else x * (ceiling / p)


def rms_normalize(x, target_db: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cur = rms(x)
    if cur < 1e-9:
        return x
    return x * (db_to_gain(target_db) / cur)


def loudness_match(x, reference, ceiling: float = 0.99) -> np.ndarray:
    """Match RMS to a reference so A/B comparisons are level-fair.

    Level matching is intentional: without it, a preset that simply boosts
    output would sound "better" in a shortlist A/B test.
    """
    ref_rms = rms(reference)
    y = np.asarray(x, dtype=float)
    cur = rms(y)
    if cur > 1e-9 and ref_rms > 1e-9:
        y = y * (ref_rms / cur)
    return peak_limit(y, ceiling)


# --------------------------------------------------------------------------
# WAV IO
# --------------------------------------------------------------------------


def _pcm_bytes(x, bit_depth: int) -> tuple[bytes, int]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("refusing to encode an empty signal")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or inf")
    x = np.clip(x, -1.0, 1.0)
    if bit_depth == 16:
        return np.round(x * 32767.0).astype("<i2").tobytes(), 2
    if bit_depth == 24:
        ints = np.round(x * (2**23 - 1)).astype("<i4")
        return ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes(), 3
    raise ValueError(f"unsupported bit depth: {bit_depth}")


def wav_bytes(x, sr: int = SR, bit_depth: int = 16) -> bytes:
    """Raw PCM payload exactly as ``write_wav`` would encode it."""
    del sr  # sample rate is header metadata; the payload is what is hashed
    return _pcm_bytes(x, bit_depth)[0]


def wav_file_bytes(x, sr: int = SR, bit_depth: int = 16) -> bytes:
    """A complete RIFF/WAVE file in memory, byte-identical to ``write_wav``.

    Used to hash rendered audio without touching disk, which is how runs are
    proven reproducible.
    """
    data, width = _pcm_bytes(x, bit_depth)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(int(sr))
        w.writeframes(data)
    return buf.getvalue()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_wav(path, x, sr: int = SR, bit_depth: int = 16) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_file_bytes(x, sr, bit_depth))
    return path


def read_wav(path, sr: int | None = SR) -> tuple[np.ndarray, int]:
    """Read a PCM WAV as mono float64, optionally resampling to ``sr``."""
    path = Path(path)
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        x = ints.astype(float) / float(2**23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(float) / float(2**31)
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(float) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if sr is not None and rate != sr:
        x = resample_ratio(x, sr / rate, rate)
        rate = sr
    return x, rate


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# spectral helpers
# --------------------------------------------------------------------------


def _hann(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1]  # periodic Hann, sums correctly in OLA


def lowpass_fft(x, cutoff_hz: float, sr: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return x.copy()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    # gentle 1/6-octave rolloff to avoid Gibbs ringing
    with np.errstate(divide="ignore"):
        t = np.log2(np.maximum(f, 1e-9) / cutoff_hz) * 6.0
    X = X * (1.0 / (1.0 + np.exp(np.clip(t, -60, 60))))
    return np.fft.irfft(X, x.size)


def resample_ratio(x, ratio: float, sr: int = SR) -> np.ndarray:
    """Resample by ``ratio`` (output length = len(x) * ratio)."""
    x = np.asarray(x, dtype=float)
    if abs(ratio - 1.0) < 1e-9 or x.size == 0:
        return x.copy()
    if ratio < 1.0:
        # anti-alias before decimating
        x = lowpass_fft(x, 0.45 * sr * ratio, sr)
    n_out = max(2, int(round(x.size * ratio)))
    src = np.arange(x.size)
    dst = np.linspace(0.0, x.size - 1.0, n_out)
    return np.interp(dst, src, x)


def resample_to(x, sr_in: int, sr_out: int) -> np.ndarray:
    return resample_ratio(x, sr_out / sr_in, sr_in)


@dataclass
class STFT:
    spec: np.ndarray  # (frames, bins) complex
    n_fft: int
    hop: int
    offset: int  # samples of zero padding before the signal

    @property
    def freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, 1.0 / SR)


def stft(x, n_fft: int = 1024, hop: int = 256, pad: bool = True) -> STFT:
    x = np.asarray(x, dtype=float).reshape(-1)
    offset = n_fft if pad else 0
    xp = np.concatenate([np.zeros(offset), x, np.zeros(n_fft)])
    n_frames = 1 + (xp.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * _hann(n_fft)
    return STFT(np.fft.rfft(frames, axis=1), n_fft, hop, offset)


def istft(s: STFT, length: int | None = None) -> np.ndarray:
    win = _hann(s.n_fft)
    frames = np.fft.irfft(s.spec, s.n_fft, axis=1) * win
    n_frames = frames.shape[0]
    out = np.zeros((n_frames - 1) * s.hop + s.n_fft)
    wsum = np.zeros_like(out)
    w2 = win * win
    for i in range(n_frames):
        a = i * s.hop
        out[a : a + s.n_fft] += frames[i]
        wsum[a : a + s.n_fft] += w2
    y = out / np.maximum(wsum, 1e-8)
    y = y[s.offset :]
    if length is not None:
        if y.size < length:
            y = np.pad(y, (0, length - y.size))
        y = y[:length]
    return y


def apply_spectral_gain(x, gain_fn, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Multiply each STFT bin by ``gain_fn(freqs, frame_index)``."""
    s = stft(x, n_fft, hop)
    freqs = s.freqs
    n_frames = s.spec.shape[0]
    # gain_fn may be vectorised over frames; evaluate per frame for safety
    gains = np.empty_like(s.spec.real)
    for i in range(n_frames):
        gains[i] = gain_fn(freqs, i)
    return istft(STFT(s.spec * gains, s.n_fft, s.hop, s.offset), len(x))


# --------------------------------------------------------------------------
# time-domain effects
# --------------------------------------------------------------------------


def drive(x, amount: float) -> np.ndarray:
    """tanh saturation, level-aware so ``amount`` means the same thing at any
    input level: the signal is brought to -1 dBFS of drive headroom first and
    the original peak is restored afterwards."""
    x = np.asarray(x, dtype=float)
    if amount <= 0:
        return x.copy()
    level = peak(x)
    if level < 1e-9:
        return x.copy()
    k = 1.0 + 9.0 * float(amount)
    driven = np.tanh(k * (x / level) * 0.9) / np.tanh(k)
    return driven * (level / 0.9)


def ring_mod(x, freq_hz: float, amount: float, sr: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if amount <= 0:
        return x.copy()
    t = np.arange(x.size) / sr
    mod = np.sin(2.0 * np.pi * freq_hz * t)
    return (1.0 - amount) * x + amount * (x * mod)


def tremolo(x, rate_hz: float, depth: float, sr: int = SR, phase: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if depth <= 0 or rate_hz <= 0:
        return x.copy()
    t = np.arange(x.size) / sr
    lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * rate_hz * t + phase))
    return x * (1.0 - depth + depth * lfo)


def comb(x, delay_ms: float, mix: float, feedback: float = 0.35, sr: int = SR) -> np.ndarray:
    """Feed-forward + feedback comb. Feedback is computed block-wise so the
    recursion stays vectorised."""
    x = np.asarray(x, dtype=float)
    d = max(1, int(round(delay_ms * 1e-3 * sr)))
    if mix <= 0 or d >= x.size:
        return x.copy()
    wet = x.copy()
    for start in range(d, x.size, d):
        end = min(start + d, x.size)
        wet[start:end] += feedback * wet[start - d : start - d + (end - start)]
    return (1.0 - mix) * x + mix * wet


def crush(x, bits: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if bits >= 16:
        return x.copy()
    levels = 2.0 ** max(1, int(bits)) / 2.0
    return np.round(x * levels) / levels


def noise_bed(
    n: int, amount: float, sr: int = SR, rng=None, colour: float = 0.5
) -> np.ndarray:
    """Decorrelated noise used as an air/texture layer.

    ``colour`` tilts the spectrum instead of only high-passing it: ``0`` is
    dark (about -6 dB/octave, pink-ish), ``0.5`` is flat white and ``1`` is a
    bright hiss (about +6 dB/octave). Rumble below 120 Hz is always removed, so
    the bed reads as texture rather than as a low-frequency rumble.
    """
    if amount <= 0:
        return np.zeros(n)
    rng = np.random.default_rng(0) if rng is None else rng
    nz = rng.standard_normal(n)
    nz = nz - lowpass_fft(nz, 120.0, sr)
    tilt = (float(min(1.0, max(0.0, colour))) - 0.5) * 12.0  # dB/octave
    if abs(tilt) > 0.01 and n >= 8:
        X = np.fft.rfft(nz)
        f = np.maximum(np.fft.rfftfreq(n, 1.0 / sr), 20.0)
        X = X * 10.0 ** (tilt * np.log2(f / 1000.0) / 20.0)
        nz = np.fft.irfft(X, n)
    nz = normalize(nz, 1.0)
    return nz * amount


def fade(x, ms_in: float = 5.0, ms_out: float = 15.0, sr: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    a = min(x.size // 2, int(ms_in * 1e-3 * sr))
    b = min(x.size // 2, int(ms_out * 1e-3 * sr))
    if a > 0:
        x[:a] *= np.linspace(0.0, 1.0, a)
    if b > 0:
        x[-b:] *= np.linspace(1.0, 0.0, b)
    return x
