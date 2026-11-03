import numpy as np
import pytest

from vox import color, dsp, generate, intelligibility as ig, presets


@pytest.fixture(scope="module")
def phrase():
    return generate.phrase_reference()


def assess_with(phrase, **overrides):
    params = dict(presets.NEUTRAL)
    params.update(overrides)
    out = dsp.loudness_match(color.render(phrase, params), phrase)
    return ig.assess(phrase, out)


def test_band_weights_are_normalised():
    assert len(ig.BAND_CENTERS) == len(ig.BAND_WEIGHTS)
    assert abs(float(np.sum(ig.BAND_WEIGHTS)) - 1.0) < 1e-12


def test_identity_is_fully_intelligible(phrase):
    a = ig.assess(phrase, phrase)
    assert a.sti > 0.999
    assert a.wer < 0.01
    assert a.passes


def test_uniform_gain_is_not_a_defect(phrase):
    for gain in (0.25, 0.5, 2.0):
        a = ig.assess(phrase, phrase * gain)
        assert a.wer < 0.01, a.summary()


def test_delay_is_not_a_defect(phrase):
    delayed = np.concatenate([np.zeros(480), phrase])[: phrase.size]
    a = ig.assess(phrase, delayed)
    assert a.wer < 0.01, a.summary()


def test_mild_eq_passes(phrase):
    a = assess_with(phrase, low_shelf_db=2.0, high_shelf_db=-2.0, peak_db=1.5)
    assert a.passes, a.summary()


def test_formant_extremes_degrade_intelligibility(phrase):
    baseline = assess_with(phrase)
    for ratio in (0.70, 1.45):
        a = assess_with(phrase, formant_shift=ratio)
        assert a.wer > baseline.wer
        assert a.sti < baseline.sti
        assert a.bands and a.worst_band.transmission < 1.0


def test_loud_noise_bed_is_refused(phrase):
    assert not assess_with(phrase, noise_mix=1.0).passes
    assert not assess_with(phrase, noise_mix=2.0).passes


def test_extreme_pitch_down_is_refused(phrase):
    assert not assess_with(phrase, pitch_semitones=-9.0).passes


def test_bounds_tail_combo_is_refused(phrase):
    a = assess_with(
        phrase,
        pitch_semitones=3.4,
        formant_shift=1.13,
        drive=0.38,
        noise_mix=0.06,
        comb_mix=0.3,
    )
    assert not a.passes, a.summary()


def test_stacked_extremes_are_refused(phrase):
    a = assess_with(
        phrase,
        pitch_semitones=6.0,
        formant_shift=1.28,
        drive=0.6,
        noise_mix=0.2,
        high_shelf_db=-9.0,
    )
    assert not a.passes, a.summary()


def test_gentle_effects_pass(phrase):
    for kwargs in (
        {"tremolo_depth": 0.35, "tremolo_hz": 5.0},
        {"comb_mix": 0.3, "comb_delay_ms": 6.0},
        {"crush_bits": 10.0},
        {"pitch_semitones": 2.0},
        {"formant_shift": 1.13},
    ):
        a = assess_with(phrase, **kwargs)
        assert a.passes, f"{kwargs}: {a.summary()}"


def test_wer_monotonic_in_noise(phrase):
    wers = [assess_with(phrase, noise_mix=n).wer for n in (0.0, 0.1, 0.2, 0.35)]
    assert wers == sorted(wers), wers


def test_noise_bed_reduces_band_snr(phrase):
    quiet = assess_with(phrase)
    noisy = assess_with(phrase, noise_mix=0.25)
    assert min(b.snr_db for b in noisy.bands) < min(b.snr_db for b in quiet.bands)


def test_too_short_raises():
    with pytest.raises(ValueError):
        ig.assess(np.zeros(10), np.zeros(10))


def test_word_error_rate():
    phrase = "the quick brown fox jumps over the lazy dog"
    assert ig.word_error_rate(phrase, phrase) == 0.0
    assert ig.word_error_rate(phrase, "the quick brown fox jumps over the lazy dog.") == 0.0
    assert abs(ig.word_error_rate(phrase, "the quick brown fox jumps over the lazy cat") - 1 / 9) < 1e-9
    assert ig.word_error_rate(phrase, "") == 1.0
    assert ig.word_error_rate("", "") == 0.0


def test_assessment_reports_weakest_band(phrase):
    a = assess_with(phrase, formant_shift=1.3)
    assert a.worst_band.transmission <= min(b.transmission for b in a.bands) + 1e-12
    assert "est WER" in a.summary()
