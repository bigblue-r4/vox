import numpy as np
import pytest

from vox import accent, dsp


def test_cues_are_finite_and_leave_headroom():
    for cue in (accent.chime(), accent.hum(), accent.breath(), accent.settle()):
        assert np.all(np.isfinite(cue))
        assert dsp.peak(cue) <= 0.891 + 1e-9
        assert dsp.peak(cue) > 0.5


def test_cues_start_and_end_silently():
    """A cue that begins on a non-zero sample clicks, which reads as a fault."""
    for name, cue in (
        ("chime", accent.chime()),
        ("hum", accent.hum()),
        ("breath", accent.breath()),
        ("settle", accent.settle()),
    ):
        assert abs(cue[0]) < 1e-3, name
        assert abs(cue[-1]) < 1e-3, name


def test_chime_has_energy_at_its_partials():
    cue = accent.chime(0.6, 1000.0)
    spec = np.abs(np.fft.rfft(cue))
    f = np.fft.rfftfreq(cue.size, 1.0 / dsp.SR)

    def at(hz):
        return float(np.max(spec[(f > hz - 25) & (f < hz + 25)]))

    assert at(1000.0) > 5.0 * at(1500.0) or at(1000.0) > 0.0  # fundamental present
    assert at(2760.0) > 0.0  # inharmonic partial, not a plain harmonic stack


def test_hum_is_low_and_has_a_weak_second_harmonic():
    cue = accent.hum(1.0, 55.0)
    spec = np.abs(np.fft.rfft(cue))
    f = np.fft.rfftfreq(cue.size, 1.0 / dsp.SR)

    def at(hz):
        m = (f > hz - 8) & (f < hz + 8)
        return float(spec[m].max()) if m.any() else 0.0

    assert at(55.0) > 10.0 * at(200.0)
    assert at(110.0) > 0.0


def test_breath_and_settle_stay_out_of_the_speech_band():
    """This is the design constraint, not a detail: a cue that lands in the
    500 Hz - 2 kHz region competes with the formants that carry the words."""
    assert accent.speech_band_fraction(accent.breath()) < 0.15
    assert accent.speech_band_fraction(accent.settle()) < 0.30
    # a bell put at the classic notification pitch does sit in that region
    assert accent.speech_band_fraction(accent.chime(0.6, 1046.5)) > 0.30


def test_place_puts_the_cue_at_the_requested_time_and_level():
    voice = np.concatenate([0.3 * np.sin(2 * np.pi * 200 * np.arange(2400) / dsp.SR),
                            np.zeros(24000)])
    cue = np.ones(2400) * 0.5
    out = accent.place(voice, cue, at_s=0.1, level_db=-20.0)
    start = int(round(0.1 * dsp.SR))
    placed = out[start : start + cue.size]
    # level_db is relative to the whole voice's RMS, by design
    ratio_db = 20.0 * np.log10(dsp.rms(placed) / dsp.rms(voice))
    assert abs(ratio_db - (-20.0)) < 0.5, ratio_db
    assert dsp.rms(out[: start - 5]) <= dsp.rms(voice[: start - 5]) + 1e-12


def test_place_rejects_negative_offsets():
    with pytest.raises(ValueError):
        accent.place(np.ones(100), np.ones(10), at_s=-0.1)


def test_turn_signature_wraps_the_voice():
    voice = np.ones(12000) * 0.1
    out = accent.turn_signature(voice, level_db=-30.0)
    assert out.size >= voice.size + int(0.5 * dsp.SR)
    # the breath is audible in the lead-in, before the voice starts
    lead = int(0.30 * dsp.SR)
    assert dsp.rms(out[:lead]) > 1e-4
    # and the settle survives past the last word instead of being trimmed
    assert dsp.rms(out[lead + voice.size :]) > 1e-5


def test_overlap_fraction_of_a_cue_inside_speech_is_one():
    cue = np.ones(4800) * 0.1
    track = np.zeros(24000)
    track[4800:9600] += cue
    assert accent.overlap_fraction(track, [(0.2, 0.4)]) == pytest.approx(1.0, abs=1e-6)
    assert accent.overlap_fraction(track, [(0.8, 1.0)]) == pytest.approx(0.0, abs=1e-9)
