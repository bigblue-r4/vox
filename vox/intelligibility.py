"""Estimated speech intelligibility for a colored signal.

The guard is a simplified Speech Transmission Index: the clean and processed
signals are split into octave bands, their intensity envelopes are extracted,
and the correlation between clean and processed envelopes is taken as the
apparent modulation transfer per band. Each band is penalised twice more:

* coloration — its long-term level may not depart from the clean level
  (relative to the overall gain the preset applied), because moving energy out
  of a band removes speech cues from it;
* noise — a band's effective SNR (95th percentile speech level over the 10th
  percentile noise floor) is mapped onto the STI transmission scale, so an
  added hiss bed or a raised quantisation floor lowers the index.

    TI_i = r_i * exp(-0.5 * (dev_i_dB / 5)^2) * min(1, snr_ti_i / snr_ti_clean_i)
    STI  = sum(w_i * TI_i)
    WER  = 1 / (1 + exp(10.5 * (STI - 0.50)))

Calibration anchors (logistic chosen to sit on these points):

    STI 0.95 -> ~0.9% WER   STI 0.80 -> ~4%    STI 0.70 -> ~11%
    STI 0.60 -> ~25%        STI 0.50 -> 50%

These anchors follow the shape of published STI/WER relations for sentences but
this is an *estimator*, not a measurement. ``--asr`` can run a real ASR pass
when a Whisper CLI is installed; see README "Guard limitations".
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dsp

SR = dsp.SR

#: Octave-band centres and IEC 60268-16 style weights, renormalised to 7 bands.
BAND_CENTERS = (250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0)
BAND_WEIGHTS = np.array([0.085, 0.127, 0.148, 0.157, 0.152, 0.147, 0.133])
BAND_WEIGHTS = BAND_WEIGHTS / BAND_WEIGHTS.sum()

ENV_LOWPASS_HZ = 25.0
ENV_RATE = 200.0
WER_THRESHOLD = 0.05
K = 10.5
STI_MIDPOINT = 0.50


@dataclass
class BandResult:
    center_hz: float
    correlation: float
    level_dev_db: float
    snr_db: float
    transmission: float


@dataclass
class Assessment:
    sti: float
    wer: float
    bands: list[BandResult]

    @property
    def passes(self) -> bool:
        return self.wer <= WER_THRESHOLD

    @property
    def worst_band(self) -> BandResult:
        return min(self.bands, key=lambda b: b.transmission)

    def summary(self) -> str:
        w = self.worst_band
        return (
            f"STI {self.sti:.3f} · est WER {self.wer * 100:.1f}% "
            f"(weakest band {w.center_hz:.0f} Hz, TI {w.transmission:.2f})"
        )


def _bandpass(x: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    n = x.size
    if n < 32:
        return x.copy()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    # soft octave-ish skirts so adjacent bands overlap like a real filterbank
    with np.errstate(divide="ignore"):
        above = 1.0 / (1.0 + np.exp(np.clip(np.log2(np.maximum(f, 1e-9) / hi) * 8.0, -60, 60)))
        below = 1.0 / (1.0 + np.exp(np.clip(np.log2(lo / np.maximum(f, 1e-9)) * 8.0, -60, 60)))
    return np.fft.irfft(X * above * below, n)


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    env = np.abs(x)
    env = dsp.lowpass_fft(env, ENV_LOWPASS_HZ, sr)
    step = max(1, int(round(sr / ENV_RATE)))
    env = env[::step]
    env = env - env.mean()
    n = env.size
    return env if n < 4 else env / (np.std(env) + 1e-12)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 8:
        return 0.0
    r = float(np.corrcoef(a[:n], b[:n])[0, 1])
    if not np.isfinite(r):
        return 0.0
    return max(0.0, min(1.0, r))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-18))


def _snr_db(x: np.ndarray, sr: int, block_ms: float = 20.0) -> float:
    """Speech-to-floor SNR from a 20 ms short-time energy track.

    The 95th percentile is the speech level, the 10th percentile the noise
    floor. Works on band signals, so a hiss bed or quantisation floor in a band
    lowers that band's SNR even when its long-term level barely moves.
    """
    block = max(8, int(block_ms * 1e-3 * sr))
    n = (x.size // block) * block
    if n < block * 8:
        return 60.0
    energy = np.square(x[:n]).reshape(-1, block).mean(axis=1) + 1e-18
    speech = 10.0 * np.log10(np.percentile(energy, 95))
    floor = 10.0 * np.log10(np.percentile(energy, 10))
    return float(max(0.0, min(60.0, speech - floor)))


def _best_lag(clean: np.ndarray, processed: np.ndarray, sr: int, max_ms: float = 30.0) -> int:
    """Broadband envelope lag (in envelope samples) that best aligns clean
    and processed. Global delay is not an intelligibility defect, so the
    estimator is allowed to time-align before correlating bands."""
    ce = _envelope(dsp.lowpass_fft(clean, 1500.0, sr), sr)
    pe = _envelope(dsp.lowpass_fft(processed, 1500.0, sr), sr)
    n = min(ce.size, pe.size)
    if n < 16:
        return 0
    c = np.correlate(pe[:n], ce[:n], "full")
    centre = n - 1
    radius = int(round(max_ms * 1e-3 * ENV_RATE))
    lo = max(0, centre - radius)
    hi = min(c.size, centre + radius + 1)
    return int(np.argmax(c[lo:hi]) + lo - centre)


def _shift(a: np.ndarray, lag: int) -> np.ndarray:
    """Return ``a`` advanced by ``lag`` samples (negative = delayed)."""
    if lag == 0:
        return a
    if lag > 0:
        return a[lag:]
    return np.concatenate([np.zeros(-lag), a])


def assess(clean, processed, sr: int = SR) -> Assessment:
    """Estimate STI and WER for ``processed`` relative to ``clean``."""
    clean = np.asarray(clean, dtype=float).reshape(-1)
    processed = np.asarray(processed, dtype=float).reshape(-1)
    n = min(clean.size, processed.size)
    if n < 256:
        raise ValueError("signal too short to assess (need >= 256 samples)")
    clean, processed = clean[:n], processed[:n]
    lag = _best_lag(clean, processed, sr)
    # A uniform level change is not an intelligibility defect: measure band
    # deviation relative to the overall gain the preset applied.
    overall_db = 20.0 * np.log10((_rms(processed) + 1e-12) / (_rms(clean) + 1e-12))

    bands: list[BandResult] = []
    for center, weight in zip(BAND_CENTERS, BAND_WEIGHTS):
        lo, hi = center / np.sqrt(2.0), center * np.sqrt(2.0)
        cb = _bandpass(clean, lo, hi, sr)
        pb = _bandpass(processed, lo, hi, sr)
        r = _corr(_envelope(cb, sr), _shift(_envelope(pb, sr), lag))
        dev_db = abs(
            20.0 * np.log10((_rms(pb) + 1e-12) / (_rms(cb) + 1e-12)) - overall_db
        )
        penalty = float(np.exp(-0.5 * (dev_db / 5.0) ** 2))
        snr = _snr_db(pb, sr)
        snr_clean = _snr_db(cb, sr)
        noise_ti = float(np.clip((snr + 15.0) / 30.0, 0.0, 1.0))
        noise_ti_clean = float(np.clip((snr_clean + 15.0) / 30.0, 0.0, 1.0))
        noise_ratio = float(np.clip(noise_ti / max(noise_ti_clean, 1e-6), 0.0, 1.0))
        bands.append(
            BandResult(
                center_hz=center,
                correlation=r,
                level_dev_db=dev_db,
                snr_db=snr,
                transmission=r * penalty * noise_ratio,
            )
        )

    sti = float(np.clip(np.dot(BAND_WEIGHTS, [b.transmission for b in bands]), 0.0, 1.0))
    wer = float(1.0 / (1.0 + np.exp(K * (sti - STI_MIDPOINT))))
    return Assessment(sti=sti, wer=wer, bands=bands)


def asr_available() -> str | None:
    """Return the path of a Whisper CLI if one is installed."""
    return shutil.which("whisper") or shutil.which("whisper-cli") or shutil.which("main")


def asr_wer(audio_path, phrase: str, timeout: float = 300.0) -> float | None:
    """Optional real check: transcribe with a Whisper CLI and score WER.

    Returns ``None`` (never a guess) when no CLI is available or it fails, so
    the estimator stays the default and ASR is strictly additive evidence.
    """
    exe = asr_available()
    if exe is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [exe, str(audio_path), "--language", "en", "--output_format", "txt",
                 "--output_dir", tmp, "--fp16", "False"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        outs = sorted(Path(tmp).glob("*.txt"))
        if not outs:
            return None
        hypothesis = outs[0].read_text(encoding="utf-8", errors="replace")
    return word_error_rate(phrase, hypothesis)


def _normalise_words(text: str) -> list[str]:
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return cleaned.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein WER over words, in [0, 1] unless the hypothesis is longer."""
    ref = _normalise_words(reference)
    hyp = _normalise_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if r == h else 1))
            )
        prev = cur
    return min(1.0, prev[-1] / len(ref))
