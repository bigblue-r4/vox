"""A/B shortlist page and the local tagging server.

The page is a single self-contained HTML file with the run manifest inlined so
it works from ``file://`` (no fetch of local JSON, which browsers block).
Keep/discard decisions persist in ``localStorage`` and can be exported as JSON.

``vox serve <run>`` adds the part a static page cannot do: decisions are POSTed
back, written to ``tags.json`` next to the run, and every "keep" is promoted
into ``presets/shipped/`` immediately.
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import presets

AUDITION = "audition.wav"


def _card(entry: dict) -> str:
    params = entry["params"]
    guard = entry["guard"]
    return f"""
      <article class="card" data-name="{entry['name']}" data-audition="items/{entry['name']}/{AUDITION}">
        <header>
          <h2>{entry['name']}</h2>
          <span class="badge">{entry['signature'] or 'generated'}</span>
        </header>
        <p class="meta">{entry['source'] or 'designed'} · STI {guard['sti']:.2f} · est WER {guard['est_wer'] * 100:.1f}%</p>
        <p class="params">{presets.encode(params)}</p>
        <audio controls preload="none" src="items/{entry['name']}/{AUDITION}"></audio>
        <div class="row">
          <button class="keep" data-tag="keep">Keep</button>
          <button class="discard" data-tag="discard">Discard</button>
          <button class="dry" data-tag="">Dry ref</button>
        </div>
        <p class="state" aria-live="polite">untagged</p>
      </article>"""


_CSS = """
:root { color-scheme: dark; --bg:#0d0f12; --panel:#151a20; --line:#242c36; --ink:#e8edf3; --dim:#8b98a8;
        --keep:#3ddc97; --discard:#ff6b6b; --accent:#7aa2ff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
header.top { padding:20px 24px; border-bottom:1px solid var(--line); display:flex;
             flex-wrap:wrap; gap:16px; align-items:baseline; }
h1 { font-size:16px; margin:0; letter-spacing:.08em; text-transform:uppercase; }
.dim { color:var(--dim); }
main { padding:18px 24px 64px; }
.bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:18px; }
button { background:var(--panel); color:var(--ink); border:1px solid var(--line);
         border-radius:6px; padding:7px 12px; font:inherit; cursor:pointer; }
button:hover { border-color:var(--accent); }
button.keep.on { background:var(--keep); color:#04231a; border-color:var(--keep); }
button.discard.on { background:var(--discard); color:#2a0606; border-color:var(--discard); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card.playing { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
.card.kept { border-color:var(--keep); }
.card.discarded { opacity:.55; }
.card header { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.card h2 { font-size:14px; margin:0; letter-spacing:.04em; }
.badge { font-size:11px; color:var(--dim); border:1px solid var(--line); border-radius:999px; padding:2px 8px; }
.meta { color:var(--dim); font-size:12px; margin:6px 0 4px; }
.params { color:var(--dim); font-size:11.5px; min-height:32px; margin:0 0 8px; }
audio { width:100%; height:32px; margin-bottom:8px; }
.row { display:flex; gap:8px; }
.row button { flex:1; padding:5px 8px; font-size:12px; }
.state { font-size:11px; color:var(--dim); margin:8px 0 0; }
footer { position:fixed; left:0; right:0; bottom:0; background:#0b0d10; border-top:1px solid var(--line);
         padding:10px 24px; font-size:12px; color:var(--dim); display:flex; gap:18px; flex-wrap:wrap; }
"""

_JS = """
const RUN = __RUN__;
const KEY = 'vox-tags:' + RUN.master_seed + ':' + RUN.count;
const REF = 'reference.wav';
let tags = load();
let queue = [], index = -1;

function load() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; }
}
function save() { localStorage.setItem(KEY, JSON.stringify(tags)); }
function card(name) { return document.querySelector('.card[data-name="' + CSS.escape(name) + '"]'); }

function paint() {
  document.querySelectorAll('.card').forEach(el => {
    const name = el.dataset.name, tag = tags[name];
    el.classList.toggle('kept', tag === 'keep');
    el.classList.toggle('discarded', tag === 'discard');
    el.querySelector('.keep').classList.toggle('on', tag === 'keep');
    el.querySelector('.discard').classList.toggle('on', tag === 'discard');
    el.querySelector('.state').textContent = tag || 'untagged';
  });
  const kept = Object.values(tags).filter(v => v === 'keep').length;
  document.getElementById('count').textContent =
    kept + ' kept · ' + RUN.items.length + ' generated · target ' + RUN.target_min + '-' + RUN.target_max;
}

function setTag(name, tag) {
  if (!tag) { delete tags[name]; } else { tags[name] = tag; }
  save(); paint(); post(name, tag);
}

async function post(name, tag) {
  try {
    await fetch('api/tag', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, tag }) });
    document.getElementById('save').textContent = 'saved to run';
  } catch (e) {
    document.getElementById('save').textContent = 'local only (static page)';
  }
}

function player() {
  let el = document.getElementById('player');
  if (!el) { el = new Audio(); el.id = 'player'; }
  return el;
}

function playOne(name) {
  const p = player();
  document.querySelectorAll('.card').forEach(c => c.classList.remove('playing'));
  const el = card(name); if (el) el.classList.add('playing');
  p.src = el ? el.dataset.audition : REF;
  p.play().catch(() => {});
}

function playAll() {
  queue = RUN.items.map(i => i.name);
  index = 0;
  if (queue.length) playOne(queue[0]);
}

document.getElementById('playall').onclick = playAll;
document.getElementById('next').onclick = () => {
  if (index + 1 < queue.length) { index += 1; playOne(queue[index]); }
};
document.getElementById('prev').onclick = () => {
  if (index > 0) { index -= 1; playOne(queue[index]); }
};
document.getElementById('dry').onclick = () => {
  const p = player(); p.src = REF; p.play().catch(() => {});
};
document.getElementById('export').onclick = () => {
  const payload = { master_seed: RUN.master_seed, count: RUN.count, tags };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'tags.json'; a.click();
  URL.revokeObjectURL(a.href);
};
document.getElementById('clear').onclick = () => {
  if (!confirm('Clear all tags in this browser?')) return;
  tags = {}; save(); paint();
};

player().addEventListener('ended', () => {
  if (index + 1 < queue.length) { index += 1; setTimeout(() => playOne(queue[index]), 320); }
});

document.querySelectorAll('.card').forEach(el => {
  const name = el.dataset.name;
  el.querySelector('.keep').onclick = () => setTag(name, tags[name] === 'keep' ? '' : 'keep');
  el.querySelector('.discard').onclick = () => setTag(name, tags[name] === 'discard' ? '' : 'discard');
  el.querySelector('.dry').onclick = () => {
    const p = player(); p.src = REF; p.play().catch(() => {});
  };
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const current = queue[index];
  if (e.key === 'k' && current) setTag(current, 'keep');
  if (e.key === 'd' && current) setTag(current, 'discard');
  if (e.key === 'ArrowRight') document.getElementById('next').click();
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  if (e.code === 'Space') { e.preventDefault(); const p = player(); p.paused ? p.play() : p.pause(); }
});

paint();
"""


def render_html(run_index: dict, title: str) -> str:
    cards = "".join(_card(entry) for entry in run_index["items"])
    payload = {
        "master_seed": run_index["master_seed"],
        "count": run_index["count"],
        "target_min": 6,
        "target_max": 8,
        "items": [{"name": i["name"]} for i in run_index["items"]],
    }
    js = _JS.replace("__RUN__", json.dumps(payload))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style></head>
<body>
<header class="top">
  <h1>vox shortlist</h1>
  <span class="dim">seed {run_index['master_seed']} · {run_index['count']} presets ·
    guard est WER ≤ {run_index['guard']['threshold'] * 100:.0f}% · phrase “{run_index['phrase']}”</span>
</header>
<main>
  <div class="bar">
    <button id="playall">▶ Play all back to back</button>
    <button id="prev">◀ Prev</button>
    <button id="next">Next ▶</button>
    <button id="dry">A/B dry reference</button>
    <button id="export">Export tags.json</button>
    <button id="clear">Clear</button>
    <span class="dim">keys: k keep · d discard · ← → · space</span>
    <span class="dim" id="save"></span>
  </div>
  <div class="grid">{cards}</div>
</main>
<footer><span id="count"></span>
  <span>Kept presets are promoted to presets/shipped/ by <code>vox promote</code> or automatically under <code>vox serve</code>.</span>
</footer>
<script>{js}</script>
</body></html>
"""


def write_shortlist(run_dir, run_index: dict) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "shortlist.html"
    path.write_text(
        render_html(run_index, f"vox shortlist · seed {run_index['master_seed']}"),
        encoding="utf-8",
    )
    return path


def read_tags(run_dir) -> dict:
    path = Path(run_dir) / "tags.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_tags(run_dir, tags: dict) -> Path:
    path = Path(run_dir) / "tags.json"
    path.write_text(json.dumps(tags, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class _Handler(BaseHTTPRequestHandler):
    run_dir: Path
    server_version = "vox-shortlist"

    def log_message(self, fmt, *args):  # keep the console quiet and useful
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html", "/shortlist.html"):
            body = (self.run_dir / "shortlist.html").read_bytes()
            return self._send(200, body, "text/html; charset=utf-8")
        if path == "/api/state":
            payload = json.dumps(
                {"tags": read_tags(self.run_dir)}, sort_keys=True
            ).encode()
            return self._send(200, payload, "application/json")
        rel = path.lstrip("/")
        if rel.startswith("items/") or rel == "reference.wav":
            target = (self.run_dir / rel).resolve()
            try:
                target.relative_to(self.run_dir.resolve())
            except ValueError:
                return self._send(403, b"forbidden", "text/plain")
            if target.suffix != ".wav" or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(target.name)[0] or "audio/wav"
            return self._send(200, target.read_bytes(), ctype)
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/tag":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 65536:
            return self._send(400, b"bad request", "text/plain")
        try:
            payload = json.loads(self.rfile.read(length))
            name = str(payload["name"])
            tag = payload.get("tag")
        except (ValueError, KeyError):
            return self._send(400, b"bad request", "text/plain")
        run_index = json.loads((self.run_dir / "index.json").read_text(encoding="utf-8"))
        names = {entry["name"] for entry in run_index["items"]}
        if name not in names:
            return self._send(404, b"unknown preset", "text/plain")
        tags = read_tags(self.run_dir)
        if tag in ("keep", "discard"):
            tags[name] = tag
        elif tag in (None, ""):
            tags.pop(name, None)
        else:
            return self._send(400, b"bad tag", "text/plain")
        write_tags(self.run_dir, tags)

        promoted = None
        if tag == "keep":
            from . import promote as promote_mod

            promote_mod.promote(self.run_dir, [name])
            promoted = name
        self._send(
            200,
            json.dumps({"ok": True, "tags": tags, "promoted": promoted}).encode(),
            "application/json",
        )


def serve(run_dir, port: int = 8765, host: str = "127.0.0.1") -> None:
    """Serve the shortlist with live tagging + automatic promotion."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "shortlist.html").exists():
        raise FileNotFoundError(f"{run_dir}/shortlist.html not found")
    handler = type("Handler", (_Handler,), {"run_dir": run_dir})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"vox shortlist: {url}")
    print("keep/discard posts back to the run and promotes kept presets; Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
