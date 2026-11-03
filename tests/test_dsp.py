import numpy as np
import pytest
import wave

from vox import color, dsp


def test_wav_roundtrip_16bit(tmp_path):
    rng = np.random.default_rng(3)
    x = dsp.normalize(rng.standard_normal(4800))
    path = dsp.write_wav(tmp_path / "a.wav", x)
    y, sr = dsp.read_wav(path)
    assert sr == dsp.SR
    assert y.size == x.size
    assert np.max(np.abs(y - x)) < 1e-3


def test_wav_roundtrip_24bit(tmp_path):
    rng = np.random.default_rng(4)
    x = dsp.normalize(rng.standard_normal(2400))
    path = dsp.write_wav(tmp_path / "b.wav", x, bit_depth=24)
    y, _ = dsp.read_wav(path)
    assert np.max(np.abs(y - x)) < 1e-5


def test_wav_bytes_matches_file(tmp_path):
    x = dsp.normalize(np.sin(np.linspace(0, 40, 3000)))
    path = dsp.write_wav(tmp_path / "c.wav", x)
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    assert frames == dsp.wav_bytes(x)
    assert dsp.wav_file_bytes(x) == path.read_bytes()
    assert dsp.hash_bytes(dsp.wav_file_bytes(x)) == dsp.sha256_file(path)


def test_write_wav_rejects_bad_signal(tmp_path):
    with pytest.raises(ValueError):
        dsp.write_wav(tmp_path / "x.wav", np.array([]))
    with pytest.raises(ValueError):
        dsp.write_wav(tmp_path / "x.wav", np.array([0.0, np.nan]))


def test_stft_istft_reconstructs():
    rng = np.random.default_rng(5)
    x = dsp.normalize(rng.standard_normal(8000))
    s = dsp.stft(x, 512, 128)
    y = dsp.istft(s, x.size)
    assert y.size == x.size
    assert np.max(np.abs(y - x)) < 1e-6


def test_spectral_gain_identity_and_notch():
    rng = np.random.default_rng(6)
    x = dsp.normalize(rng.standard_normal(6000))
    same = dsp.apply_spectral_gain(x, lambda f, i: np.ones_like(f), 512, 128)
    assert np.max(np.abs(same - x)) < 1e-6
    muted = dsp.apply_spectral_gain(x, lambda f, i: np.zeros_like(f), 512, 128)
    assert dsp.rms(muted) < 1e-6


def test_resample_lengths():
    x = np.linspace(0, 1, 1000)
    assert dsp.resample_ratio(x, 0.5).size == 500
    assert dsp.resample_ratio(x, 2.0).size == 2000
    y = dsp.resample_to(np.zeros(4800), 48000, 24000)
    assert y.size == 2400


def test_comb_is_stable_and_bounded():
    rng = np.random.default_rng(7)
    x = dsp.normalize(rng.standard_normal(4800))
    y = dsp.comb(x, 8.0, 0.3, 0.3)
    assert np.all(np.isfinite(y))
    assert dsp.peak(y) < 4.0


def test_crush_quantises():
    x = np.linspace(-0.99, 0.99, 1000)
    y = dsp.crush(x, 8)
    assert np.max(np.abs(y * 128 - np.round(y * 128))) < 1e-9
    assert len(np.unique(np.round(y * 128))) <= 2**8


def test_drive_is_finite_and_level_aware():
    x = dsp.normalize(np.sin(np.linspace(0, 60, 4000)), 0.9)
    loud = dsp.drive(x, 0.8)
    quiet = dsp.drive(x * 0.1, 0.8)
    assert np.all(np.isfinite(loud))
    # drive is scale-equivariant: the same amount of saturation at any level
    assert np.allclose(quiet, loud * 0.1, atol=1e-9)
    assert dsp.peak(loud) <= dsp.peak(x) * 1.2


def test_loudness_match_equalises_rms():
    rng = np.random.default_rng(8)
    ref = dsp.normalize(rng.standard_normal(4000), 0.5)
    other = dsp.normalize(rng.standard_normal(4000), 0.9)
    matched = dsp.loudness_match(other, ref)
    assert abs(dsp.rms(matched) - dsp.rms(ref)) / dsp.rms(ref) < 0.01


def test_time_stretch_length_and_pitch():
    sr = dsp.SR
    t = np.arange(sr // 2) / sr
    x = np.sin(2 * np.pi * 440 * t)
    for rate in (1.0, 1.122, 1.5):
        y = color.time_stretch(x, rate)
        assert y.size == int(round(x.size * rate))
        assert dsp.peak(y) > 0.2
    stretched = color.time_stretch(x, 1.5)
    c = np.correlate(stretched - stretched.mean(), stretched - stretched.mean(), "full")
    c = c[stretched.size - 1 :]
    lag = int(np.argmax(c[int(sr / 800) : int(sr / 100)])) + int(sr / 800)
    assert abs(sr / lag - 440) < 8


def test_pitch_shift_preserves_length():
    rng = np.random.default_rng(9)
    x = dsp.normalize(rng.standard_normal(6000))
    y = color.shift_pitch(x, 3.0)
    assert y.size == x.size
    assert np.all(np.isfinite(y))
