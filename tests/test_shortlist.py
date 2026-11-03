import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from vox import dsp, presets, promote, shortlist


def make_run(tmp_path, names=("unit-a", "unit-b")):
    run_dir = tmp_path / "runs" / "seed-000001"
    (run_dir / "items").mkdir(parents=True)
    items = []
    for i, name in enumerate(names):
        item_dir = run_dir / "items" / name
        item_dir.mkdir()
        params = dict(presets.NEUTRAL, drive=0.05 * i)
        presets.Preset(name=name, params=params, source="fm-bleep",
                       signature="grit", seed={"master": 1, "index": i}).save(
            item_dir / "preset.json"
        )
        dsp.write_wav(item_dir / "sample.wav", np.zeros(dsp.SR // 2) + 0.01)
        dsp.write_wav(item_dir / "audition.wav", np.zeros(dsp.SR // 2) + 0.01)
        (item_dir / "meta.json").write_text("{}")
        items.append({
            "name": name, "index": i, "source": "fm-bleep", "signature": "grit",
            "params": presets.round_params(params),
            "guard": {"sti": 0.95, "est_wer": 0.01},
            "dir": f"items/{name}",
        })
    index = {
        "schema": "vox.run/1", "master_seed": 1, "count": len(names), "max_attempts": 30,
        "phrase": "the quick brown fox jumps over the lazy dog",
        "guard": {"threshold": 0.05, "sti_midpoint": 0.5, "logistic_k": 10.5},
        "stats": {"accepted": len(names), "rejected": 0}, "items": items,
    }
    (run_dir / "index.json").write_text(json.dumps(index))
    shortlist.write_shortlist(run_dir, index)
    return run_dir, index


def test_page_contains_cards(tmp_path):
    run_dir, index = make_run(tmp_path)
    html = (run_dir / "shortlist.html").read_text()
    assert "unit-a" in html and "unit-b" in html
    assert html.count("<audio") == 2
    assert "items/unit-a/audition.wav" in html


def test_tags_roundtrip(tmp_path):
    run_dir, _ = make_run(tmp_path)
    assert shortlist.read_tags(run_dir) == {}
    shortlist.write_tags(run_dir, {"unit-a": "keep"})
    assert shortlist.read_tags(run_dir) == {"unit-a": "keep"}


def test_promote_and_status(tmp_path, monkeypatch):
    run_dir, _ = make_run(tmp_path)
    shipped = tmp_path / "presets" / "shipped"
    designed = tmp_path / "presets" / "designed"
    designed.mkdir(parents=True)
    promoted = promote.promote(run_dir, ["unit-a"], shipped)
    assert promoted == ["unit-a"]
    assert (shipped / "unit-a" / "preset.json").is_file()
    assert (shipped / "unit-a" / "audition.wav").is_file()
    # idempotent
    promote.promote(run_dir, ["unit-a"], shipped)
    index = json.loads((shipped / "index.json").read_text())
    assert [p["name"] for p in index["presets"]] == ["unit-a"]
    state = promote.status(shipped)
    assert state["generated"] == 1 and state["designed"] == 0
    assert state["state"] == "under"


def test_promote_rejects_unknown_name(tmp_path):
    run_dir, _ = make_run(tmp_path)
    with pytest.raises(KeyError):
        promote.promote(run_dir, ["nope"], tmp_path / "shipped")


def serve(run_dir):
    handler = type("Handler", (shortlist._Handler,), {"run_dir": run_dir.resolve()})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_server_tags_and_promotes(tmp_path, monkeypatch):
    run_dir, _ = make_run(tmp_path)
    monkeypatch.chdir(tmp_path)  # promotion writes ./presets/shipped
    httpd, base = serve(run_dir)
    try:
        with urllib.request.urlopen(base + "/api/state", timeout=5) as r:
            assert json.loads(r.read())["tags"] == {}
        req = urllib.request.Request(
            base + "/api/tag",
            data=json.dumps({"name": "unit-b", "tag": "keep"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read())
        assert body["ok"] and body["promoted"] == "unit-b"
        assert shortlist.read_tags(run_dir) == {"unit-b": "keep"}
        assert (tmp_path / "presets" / "shipped" / "unit-b" / "preset.json").is_file()

        audio = urllib.request.urlopen(base + "/items/unit-b/audition.wav", timeout=5)
        assert audio.status == 200
        audio.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_rejects_bad_requests(tmp_path):
    run_dir, _ = make_run(tmp_path)
    httpd, base = serve(run_dir)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/index.json", timeout=5)
        assert exc.value.code == 404
        req = urllib.request.Request(
            base + "/api/tag",
            data=json.dumps({"name": "nope", "tag": "keep"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 404
        req = urllib.request.Request(
            base + "/api/tag",
            data=json.dumps({"name": "unit-a", "tag": "banana"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
