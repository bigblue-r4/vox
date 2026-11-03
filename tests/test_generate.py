import json

import numpy as np
import pytest

from vox import dsp, generate, presets


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    return generate.generate(count=4, seed=4242, out_root=out)


def test_run_artifacts(run):
    assert run.run_dir.is_dir()
    assert run.accepted == 4
    assert (run.run_dir / "index.json").is_file()
    assert (run.run_dir / "seeds.jsonl").is_file()
    assert (run.run_dir / "reference.wav").is_file()
    assert (run.run_dir / "shortlist.html").is_file()
    assert (run.run_dir / "tags.json").is_file()
    index = generate.load_run(run.run_dir)
    assert index["schema"] == generate.RUN_SCHEMA
    assert index["master_seed"] == 4242
    assert index["count"] == 4
    assert len(index["items"]) == 4


def test_each_item_has_paired_files_and_metadata(run):
    for entry in generate.load_run(run.run_dir)["items"]:
        item_dir = run.run_dir / entry["dir"]
        for fname in ("sample.wav", "audition.wav", "preset.json", "meta.json"):
            assert (item_dir / fname).is_file(), f"{entry['name']}/{fname}"
        meta = json.loads((item_dir / "meta.json").read_text())
        assert meta["guard"]["passes"] is True
        assert meta["guard"]["est_wer"] <= 0.05
        assert meta["seed"]["master"] == 4242
        assert meta["hashes"]["audition_sha256"] == dsp.sha256_file(item_dir / "audition.wav")
        preset = presets.Preset.load(item_dir / "preset.json")
        assert preset.name == entry["name"]
        assert preset.signature == entry["signature"]
        sample, sr = dsp.read_wav(item_dir / "sample.wav")
        assert sr == dsp.SR and sample.size > dsp.SR


def test_guard_holds_and_sources_are_balanced(run):
    index = generate.load_run(run.run_dir)
    for entry in index["items"]:
        assert entry["guard"]["est_wer"] <= 0.05
        assert entry["guard"]["sti"] > 0.5
    sources = [e["source"] for e in index["items"]]
    assert len(set(sources)) == len(sources), "first four items should use four distinct sources"


def test_seeds_are_logged(run):
    lines = generate.read_seeds(run.run_dir)
    assert len(lines) >= 4
    accepted = [line for line in lines if line["status"] == "accepted"]
    assert len(accepted) == 4
    for line in lines:
        assert line["master_seed"] == 4242
        assert 0 <= line["index"] < 4
        assert line["attempt"] >= 1
        presets.validate(line["params"])
        if line["status"] == "rejected":
            assert line["guard"]["est_wer"] > 0.05


def test_shortlist_lists_every_preset(run):
    html = (run.run_dir / "shortlist.html").read_text(encoding="utf-8")
    for entry in generate.load_run(run.run_dir)["items"]:
        assert entry["name"] in html
        assert f"items/{entry['name']}/audition.wav" in html
    assert "reference.wav" in html
    assert "Play all back to back" in html
    assert "Keep" in html and "Discard" in html


def test_verify_run_is_reproducible(run):
    ok, problems = generate.verify_run(run.run_dir)
    assert ok, problems


def test_same_seed_reproduces_identical_audio(run, tmp_path):
    again = generate.generate(count=4, seed=4242, out_root=tmp_path)
    assert again.master_seed == run.master_seed
    for a, b in zip(run.items, again.items):
        assert a.name == b.name
        assert presets.round_params(a.params) == presets.round_params(b.params)
        pa = run.run_dir / "items" / a.name / "audition.wav"
        pb = again.run_dir / "items" / b.name / "audition.wav"
        assert dsp.sha256_file(pa) == dsp.sha256_file(pb)


def test_different_seed_changes_presets(run, tmp_path):
    other = generate.generate(count=4, seed=99, out_root=tmp_path)
    assert presets.round_params(other.items[0].params) != presets.round_params(run.items[0].params)


def test_existing_run_requires_force(run, tmp_path):
    with pytest.raises(FileExistsError):
        generate.generate(count=2, seed=4242, out_root=run.run_dir.parent)
    forced = generate.generate(count=2, seed=4242, out_root=run.run_dir.parent, overwrite=True)
    assert forced.accepted == 2


def test_count_validation(tmp_path):
    with pytest.raises(ValueError):
        generate.generate(count=0, seed=1, out_root=tmp_path)
