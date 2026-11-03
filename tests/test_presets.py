import json

import numpy as np
import pytest

from vox import presets


def test_bounds_table_is_coherent():
    for key, (lo, hi, mode) in presets.PARAMS.items():
        assert lo < hi, key
        assert mode in {"linear", "log"}, key
    assert set(presets.NEUTRAL) == set(presets.PARAMS)
    assert set(presets._JITTER) == set(presets.PARAMS)
    for sig, pinned in presets.SIGNATURES.items():
        if sig == "wildcard":
            assert pinned == {}
            continue
        assert pinned, sig
        for key, (lo, hi) in pinned.items():
            hard_lo, hard_hi, _ = presets.PARAMS[key]
            assert hard_lo <= lo < hi <= hard_hi, f"{sig}.{key} outside PARAMS"


def test_draw_colors_stays_in_bounds():
    rng = np.random.default_rng(21)
    seen_signatures = set()
    for _ in range(600):
        signature, params = presets.draw_colors(rng)
        seen_signatures.add(signature)
        assert signature in presets.SIGNATURES
        presets.validate(params)
        for key, (lo, hi, _mode) in presets.PARAMS.items():
            assert lo - 1e-9 <= params[key] <= hi + 1e-9
        assert set(params) == set(presets.PARAMS)
    assert seen_signatures == set(presets.SIGNATURES)


def test_draw_is_reproducible():
    a = presets.draw_colors(np.random.default_rng(99))
    b = presets.draw_colors(np.random.default_rng(99))
    assert a == b


def test_validate_rejects_bad_values():
    good = dict(presets.NEUTRAL)
    presets.validate(good)
    bad = dict(good, pitch_semitones=99)
    with pytest.raises(presets.PresetError):
        presets.validate(bad)
    bad = dict(good)
    bad.pop("comb_mix")
    with pytest.raises(presets.PresetError):
        presets.validate(bad)
    with pytest.raises(presets.PresetError):
        presets.validate(dict(good, drive=float("nan")))


def test_save_load_roundtrip(tmp_path):
    params = dict(presets.NEUTRAL, drive=0.2, peak_freq=2400.0)
    p = presets.Preset(name="unit-test", params=params, seed={"master": 1, "index": 2},
                       signature="grit", source="fm-bleep")
    path = p.save(tmp_path / "p.json")
    back = presets.Preset.load(path)
    assert back.name == "unit-test"
    assert back.signature == "grit"
    assert back.seed == {"master": 1, "index": 2}
    assert back.params == presets.round_params(params)


def test_from_dict_rejects_unknown_and_missing(tmp_path):
    raw = presets.Preset(name="x", params=dict(presets.NEUTRAL)).to_dict()
    presets.Preset.from_dict(raw)
    with pytest.raises(presets.PresetError):
        presets.Preset.from_dict(dict(raw, params={**raw["params"], "bogus": 1.0}))
    incomplete = dict(raw["params"])
    incomplete.pop("drive")
    with pytest.raises(presets.PresetError):
        presets.Preset.from_dict(dict(raw, params=incomplete))
    with pytest.raises(presets.PresetError):
        presets.Preset.from_dict(dict(raw, schema="vox.preset/999"))


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(presets.PresetError):
        presets.Preset.load(path)


def test_encode_is_readable():
    s = presets.encode(presets.NEUTRAL)
    assert "pit" in s and "fmt" in s and "eq" in s


def test_designed_presets_are_valid():
    designed = presets.load_dir("presets/designed")
    assert len(designed) == 4, "expected the four hand-designed presets"
    names = {p.name for p in designed}
    assert names == {"velvet-veil", "glasshouse", "static-bloom", "understudy"}
    for p in designed:
        assert p.origin == "designed"


def test_shipped_presets_load_from_subdirectories():
    shipped = presets.load_dir("presets/shipped")
    assert all(p.origin == "generated" for p in shipped)
    assert 6 <= len(shipped) <= 8, "shipped generated set should meet the 6-8 target"
    for p in shipped:
        presets.validate(p.params)


def test_designed_presets_pass_the_guard():
    from vox import color, dsp, generate, intelligibility

    phrase = generate.phrase_reference()
    for preset in presets.load_dir("presets/designed"):
        out = dsp.loudness_match(color.render(phrase, preset), phrase)
        assessment = intelligibility.assess(phrase, out)
        assert assessment.passes, f"{preset.name}: {assessment.summary()}"
