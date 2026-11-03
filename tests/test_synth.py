import numpy as np
import pytest

from vox import dsp, synth


def render(source, seed=11):
    return synth.synthesize(source, np.random.default_rng(seed))


@pytest.mark.parametrize("source", synth.SOURCES)
def test_source_is_deterministic(source):
    a = render(source)
    b = render(source)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("source", synth.SOURCES)
def test_source_is_well_formed(source):
    x = render(source)
    assert np.all(np.isfinite(x))
    assert dsp.peak(x) <= dsp.PEAK + 1e-6
    assert dsp.rms(x) > 0.005
    duration = x.size / dsp.SR
    assert 1.9 <= duration <= 2.9
    assert abs(float(np.mean(x))) < 0.01


@pytest.mark.parametrize("source", synth.SOURCES)
def test_source_is_seed_sensitive(source):
    a = render(source, 11)
    b = render(source, 12)
    assert not np.array_equal(a, b)


def test_sources_are_distinct():
    rendered = {s: render(s) for s in synth.SOURCES}
    for i, a in enumerate(synth.SOURCES):
        for b in synth.SOURCES[i + 1 :]:
            x, y = rendered[a], rendered[b]
            n = min(x.size, y.size)
            x = x[:n] - x[:n].mean()
            y = y[:n] - y[:n].mean()
            corr = abs(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
            assert corr < 0.5, f"{a} and {b} are too similar ({corr:.2f})"


def test_source_coverage_is_balanced():
    seen = [synth.source_for_index(i) for i in range(12)]
    assert set(seen) == set(synth.SOURCES)
    assert all(seen.count(s) == 2 for s in synth.SOURCES)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        synth.synthesize("nope", np.random.default_rng(0))


def test_every_source_has_a_blurb():
    assert set(synth.SOURCE_BLURB) == set(synth.SOURCES)
