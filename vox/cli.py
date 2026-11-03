"""``vox`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import TEST_PHRASE, __version__, dsp, presets


def _fmt_table(rows: list[tuple[str, float, float, str]]) -> str:
    out = [f"{'name':24s} {'STI':>6s} {'est WER':>8s}  signature/source"]
    for name, sti, wer, tag in rows:
        out.append(f"{name:24s} {sti:6.3f} {wer * 100:7.1f}%  {tag}")
    return "\n".join(out)


def cmd_generate(args) -> int:
    from . import generate as gen

    def progress(item, attempts):
        if not args.quiet:
            note = "" if attempts == 1 else f" ({attempts} attempts)"
            print(f"  + {item.name:22s} {item.signature:8s} STI {item.sti:.3f} "
                  f"WER {item.wer * 100:.1f}%{note}", flush=True)

    try:
        result = gen.generate(
            count=args.count,
            seed=args.seed,
            out_root=args.out,
            guard=args.guard,
            max_attempts=args.max_attempts,
            overwrite=args.force,
            progress=progress,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    index = gen.load_run(result.run_dir)
    print(f"\nrun:      {result.run_dir}")
    print(f"seed:     {result.master_seed}")
    print(f"accepted: {result.accepted}  rejected attempts: {result.rejected}")
    print(f"shortlist:{result.shortlist_path}")
    print("\nnext:")
    print(f"  vox serve {result.run_dir}")
    print(f"  vox promote {result.run_dir} --kept name1,name2")
    return 0


def cmd_serve(args) -> int:
    from . import shortlist

    try:
        shortlist.serve(args.run_dir, port=args.port)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot bind {args.port}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_promote(args) -> int:
    from . import promote as promote_mod
    from . import shortlist

    run_dir = Path(args.run_dir)
    if args.all_kept:
        tags = shortlist.read_tags(run_dir)
        names = [name for name, tag in tags.items() if tag == "keep"]
        if not names:
            print("no presets tagged 'keep' in tags.json", file=sys.stderr)
            return 1
    elif args.kept:
        names = [n.strip() for n in args.kept.split(",") if n.strip()]
    else:
        print("error: pass --kept a,b,c or --all-kept", file=sys.stderr)
        return 1
    try:
        promoted = promote_mod.promote(run_dir, names, args.shipped, force=args.force)
    except (KeyError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name in promoted:
        print(f"promoted {name} -> {Path(args.shipped) / name}")
    promote_mod.register_designed(args.designed, args.shipped)
    state = promote_mod.status(args.shipped)
    print(f"shipped: {state['generated']} generated + {state['designed']} designed "
          f"(target {state['target_min']}-{state['target_max']} generated)")
    return 0


def cmd_status(args) -> int:
    from . import promote as promote_mod

    promote_mod.register_designed(args.designed, args.shipped)
    state = promote_mod.status(args.shipped)
    print(f"generated: {state['generated']}  designed: {state['designed']}  total: {state['total']}")
    print(f"target:    {state['target_min']}-{state['target_max']} generated presets "
          f"(state: {state['state']})")
    if state["names"]:
        print("shipped:   " + ", ".join(state["names"]))
    return 0


def cmd_render(args) -> int:
    from . import color

    preset = presets.Preset.load(args.preset)
    src, sr = dsp.read_wav(args.input, dsp.SR)
    out = color.render(src, preset, sr)
    if args.loudness_match:
        out = dsp.loudness_match(out, src)
    path = dsp.write_wav(args.out, out, sr)
    print(f"wrote {path} ({out.size / sr:.2f}s, {preset.name})")
    return 0


def cmd_phrase(args) -> int:
    from . import testphrase

    path = dsp.write_wav(args.out, testphrase.synthesize(TEST_PHRASE), dsp.SR)
    print(f"wrote {path} — standard test phrase ({TEST_PHRASE!r})")
    return 0


def cmd_report(args) -> int:
    from . import generate as gen

    index = gen.load_run(args.run_dir)
    rows = [
        (
            item["name"],
            item["guard"]["sti"],
            item["guard"]["est_wer"],
            f"{item['signature'] or '-':9s} {item['source'] or 'designed'}",
        )
        for item in index["items"]
    ]
    print(f"run {args.run_dir} · seed {index['master_seed']} · {index['count']} presets")
    print(_fmt_table(rows))
    print(f"\naccepted {index['stats']['accepted']} · rejected attempts {index['stats']['rejected']}")
    return 0


def cmd_verify(args) -> int:
    from . import generate as gen

    ok, problems = gen.verify_run(args.run_dir)
    if ok:
        print(f"reproducible: every preset in {args.run_dir} rebuilds from its logged seed")
        return 0
    print("reproducibility problems:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


def cmd_preset(args) -> int:
    if args.preset_cmd == "list":
        designed = presets.load_dir(args.designed)
        shipped = presets.load_dir(args.shipped)
        if not designed and not shipped:
            print("no presets found")
            return 0
        for label, group in (("designed", designed), ("shipped", shipped)):
            for p in group:
                print(f"{label:9s} {p.name:24s} {presets.encode(p.params)}")
        return 0
    if args.preset_cmd == "validate":
        failures = 0
        for directory in (args.designed, args.shipped):
            base = Path(directory)
            for path in sorted(base.rglob("*.json")):
                if path.name in {"index.json", "meta.json"}:
                    continue
                if path.parent != base and path.name != "preset.json":
                    continue
                try:
                    p = presets.Preset.load(path)
                    print(f"ok      {path} ({p.name})")
                except presets.PresetError as exc:
                    failures += 1
                    print(f"invalid {path}: {exc}")
        return 1 if failures else 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vox",
        description="Procedural voice-color presets with an intelligibility guard.",
    )
    p.add_argument("--version", action="version", version=f"vox {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate presets + samples + auditions + shortlist")
    g.add_argument("--count", type=int, default=12, help="how many presets (default 12)")
    g.add_argument("--seed", type=int, default=None, help="master seed (default: random, logged)")
    g.add_argument("--out", default="runs", help="output root (default: runs/)")
    g.add_argument("--guard", type=float, default=0.05, help="max estimated WER (default 0.05)")
    g.add_argument("--max-attempts", type=int, default=30, help="regeneration attempts per preset")
    g.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    g.add_argument("--quiet", action="store_true")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("serve", help="serve the shortlist with live tagging and promotion")
    s.add_argument("run_dir")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    pr = sub.add_parser("promote", help="promote kept presets into presets/shipped/")
    pr.add_argument("run_dir")
    pr.add_argument("--kept", help="comma separated preset names")
    pr.add_argument("--all-kept", action="store_true", help="use tags.json keeps")
    pr.add_argument("--shipped", default="presets/shipped")
    pr.add_argument("--designed", default="presets/designed")
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(func=cmd_promote)

    st = sub.add_parser("status", help="shipped set status against the 6-8 target")
    st.add_argument("--shipped", default="presets/shipped")
    st.add_argument("--designed", default="presets/designed")
    st.set_defaults(func=cmd_status)

    r = sub.add_parser("render", help="render a preset onto an audio file")
    r.add_argument("input")
    r.add_argument("--preset", required=True)
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--loudness-match", action="store_true")
    r.set_defaults(func=cmd_render)

    t = sub.add_parser("phrase", help="write the standard test phrase WAV")
    t.add_argument("-o", "--out", default="test-phrase.wav")
    t.set_defaults(func=cmd_phrase)

    rep = sub.add_parser("report", help="print a run's presets and guard numbers")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    v = sub.add_parser("verify", help="prove a run is reproducible from its seeds")
    v.add_argument("run_dir")
    v.set_defaults(func=cmd_verify)

    pre = sub.add_parser("preset", help="inspect or validate preset files")
    pre.add_argument("preset_cmd", choices=["list", "validate"])
    pre.add_argument("--shipped", default="presets/shipped")
    pre.add_argument("--designed", default="presets/designed")
    pre.set_defaults(func=cmd_preset)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
