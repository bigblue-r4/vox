# vox

Procedural voice-colour presets with a measured intelligibility guard.

`vox` builds "colours" for a voice: a small preset of pitch, formant, EQ,
saturation and modulation values, plus the audio evidence that the colour does
not destroy intelligibility. Addendum 2 adds the generated default set:
procedurally synthesised samples, randomised presets drawn from bounded ranges,
a WER guard that regenerates anything too destructive, reproducible seeds, and
a shortlist page for A/B listening and keep/discard tagging.

Everything is synthesised in code. There are no recordings, no sample packs and
no network calls in the audio path.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 12 presets, one run directory, seeds logged
.venv/bin/python -m vox.cli generate --count 12 --seed 404

# listen, tag, promote as you go
.venv/bin/python -m vox.cli serve runs/seed-000404
```

After `pip install -e .` the same commands are available as `vox generate ...`.

## What `generate` produces

```
runs/seed-000404/
  index.json                 run manifest: seed, guard config, every preset
  seeds.jsonl                one line per attempt (accepted *and* rejected)
  reference.wav              clean test phrase, for dry A/B
  shortlist.html             self-contained A/B page with keep/discard tagging
  tags.json                  decisions (written by the tagging server)
  items/
    orchid-73/
      sample.wav             the procedurally synthesised colour sample
      audition.wav           standard test phrase through this preset
      preset.json            params + seed + signature
      meta.json              guard numbers, hashes, loudness, provenance
    ...
```

Every WAV is mono 24 kHz / 16-bit. Auditions are loudness-matched to
`reference.wav` so a preset cannot win an A/B just by being louder.

## The six sample sources

Each generated preset is paired with a fresh sample from one of six
code-only synthesis models, cycled so a run covers all of them evenly:

| source | model |
| --- | --- |
| `fm-bleep` | FM bleeps/chirps, decaying modulation index, glides |
| `noise-sweep` | band-passed noise sweeping through the spectrum |
| `modal-bell` | struck-bar/bell modal resonators, inharmonic partials, beating |
| `granular-cloud` | grain cloud of synthesized tones in a pentatonic field |
| `pulse-train` | click/pulse train through a resonant band |
| `additive-pad` | glassy additive pad, detuned partial stack |

## Presets

A preset is JSON (`schema: vox.preset/1`): a name, optional signature/source/
seed, and a complete parameter set. The render chain is fixed:

```
pitch shift (phase vocoder stretch + resample)
  -> formant shift (cepstral spectral-envelope warp)
  -> EQ (STFT low shelf / bell / high shelf)
  -> drive -> ring mod -> tremolo -> comb
  -> bit crush -> noise bed -> output gain -> peak limit
```

Bounds are the *guard-passing region*, established by measuring each
parameter's effect on the estimated WER:

| parameter | range | notes |
| --- | --- | --- |
| `pitch_semitones` | -3.5 … 3.5 | moves formants with pitch, as a tape speed change would |
| `formant_shift` | 0.90 … 1.13 | vowel-tract warp, independent of pitch |
| `low_shelf_db` | -6 … 6 | below 250 Hz |
| `high_shelf_db` | -9 … 7 | above 4.2 kHz |
| `peak_freq` | 250 … 7000 Hz | log-uniform draw |
| `peak_db` | -6 … 6 | bell gain |
| `peak_q` | 0.6 … 2.4 | bell width |
| `drive` | 0 … 0.38 | level-aware tanh |
| `ring_amount` / `ring_hz` | 0 … 0.20 / 40 … 900 Hz | |
| `tremolo_depth` / `tremolo_hz` | 0 … 0.38 / 0 … 8 Hz | |
| `comb_delay_ms` / `comb_mix` | 3 … 18 ms / 0 … 0.30 | |
| `crush_bits` | 9 … 16 | |
| `noise_mix` | 0 … 0.06 | high-passed air bed |
| `output_gain_db` | -4 … 4 | |

### Signatures

A draw picks a signature (a stylistic archetype) and then jitters the
remaining parameters around neutral. Signatures pin their own parameters to a
characteristic sub-range: `air`, `warm`, `glass`, `velvet`, `smear`, `shimmer`,
`grit`, `close`, `bright`, `static`, and `wildcard`.

`wildcard` is the generator's tail exploration: half of its parameters are
pushed to the outer 25% of their range. One draw in five is a wildcard, which is
what makes the guard's reject-and-regenerate path fire in ordinary runs
(measured: 19/40 wildcard draws pass).

## The intelligibility guard

Every candidate preset renders the fixed test phrase
(`the quick brown fox jumps over the lazy dog`) and is scored before it is
written. Anything above **5% estimated WER is rejected and regenerated**; the
refusal is logged to `seeds.jsonl` with its attempt number.

The estimator is a simplified Speech Transmission Index:

1. split clean and processed signals into 7 octave bands (250 Hz … 6 kHz);
2. time-align them on the broadband envelope (a global delay is not a defect);
3. correlate clean and processed intensity envelopes per band → `r_i`;
4. penalise coloration by the band level deviation, measured relative to the
   overall gain change;
5. penalise noise by each band's effective SNR (95th percentile speech level
   over the 10th percentile floor), mapped onto the STI transmission scale.

```
TI_i = r_i * exp(-0.5 * (dev_i_dB / 5)^2) * min(1, snr_ti_i / snr_ti_clean_i)
STI  = sum(w_i * TI_i)                       # IEC-style octave weights
WER  = 1 / (1 + exp(10.5 * (STI - 0.50)))
```

Calibration anchors of that logistic: STI 0.95 → 0.9% WER, 0.80 → 4%,
0.70 → 11%, 0.60 → 25%, 0.50 → 50%.

Measured behaviour (phrase synthesised by the bundled formant model):

| change | est WER | guard |
| --- | --- | --- |
| identity / any uniform gain / delay | 0.5% | passes |
| mild EQ, tremolo, comb, 10-bit crush, +2 st | ≤ 1% | passes |
| bounds-tail combo (pitch 3.4, formant 1.13, drive .38, noise .06, comb .3) | 6.0% | refused |
| heavy stack (pitch 6, formant 1.4, drive .8, noise 1.0, −9 dB top) | 67% | refused |
| noise bed at unity gain | 47% | refused |
| pitch −9 st | 40% | refused |

Example run (`--seed 404 --count 12`): 15 attempts, 3 refused (7.5%, 9.2%,
6.5%) and regenerated, 12 accepted at 1.2–4.2% estimated WER.

### Guard limitations

* It is an **estimator**, not a listening test. It measures envelope
  transmission, coloration and band SNR; it cannot hear harmonic distortion
  (heavy `drive`), so a saturated preset can score better than it sounds.
* The carrier is a formant-synthesis stand-in, not a human recording. It has
  real formant structure, fricatives and prosody, but it is synthetic.
* To use a real voice, render a recording through a preset directly:
  `vox render speech.wav --preset presets/shipped/orchid-73/preset.json -o out.wav`,
  then score it with any ASR. `vox.intelligibility.asr_wer()` runs a Whisper CLI
  when one is installed and returns `None` (never a guess) when it is not; the
  estimator remains the default and ASR is additive evidence.
* Extend `PARAMS` only together with a guard measurement: the bounds are a
  calibration result, not an arbitrary choice.

## Reproducibility

Each item's RNG is `SeedSequence([master_seed, index])` and every draw happens
in a fixed order, so a preset's parameters *and* its audio rebuild exactly:

```bash
vox verify runs/seed-000404
# reproducible: every preset in runs/seed-000404 rebuilds from its logged seed
```

`verify` replays every item, compares parameters and byte-compares the WAV
files. `seeds.jsonl` records every attempt with its master seed, item index,
attempt number, signature, params and guard numbers — refused draws included,
so the history of a run is auditable.

## Shortlist, tagging and promotion

`shortlist.html` is self-contained (manifest inlined, relative audio paths), so
it also works from `file://`. It plays all presets back to back, offers a dry
reference, and tags keep/discard with `k`/`d` or the buttons. Tags persist in
`localStorage` and export as `tags.json`.

`vox serve <run>` adds the round trip a static page cannot do: tags POST back to
`tags.json`, and every "keep" is promoted immediately.

```bash
vox serve runs/seed-000404                  # http://127.0.0.1:8765/
vox promote runs/seed-000404 --all-kept     # promote from tags.json
vox promote runs/seed-000404 --kept orchid-73,lumen-43
vox status                                  # 8 generated + 4 designed (target 6-8)
```

Promotion copies the preset, both WAVs and `meta.json` into
`presets/shipped/<name>/` and maintains `presets/shipped/index.json` with
origin, seed and guard numbers. Promotion is idempotent; re-promoting a name
with different parameters needs `--force`.

## The shipped set

Target: 6–8 generated presets curated from runs, plus the 4 hand-designed ones.
Currently 8 + 4 = 12:

| preset | origin | signature | source |
| --- | --- | --- | --- |
| `velvet-veil` | designed | hand-designed | — |
| `glasshouse` | designed | hand-designed | — |
| `static-bloom` | designed | hand-designed | — |
| `understudy` | designed | hand-designed | — |
| `orchid-73` | generated | close | additive-pad |
| `lumen-43` | generated | smear | pulse-train |
| `sable-11` | generated | grit | noise-sweep |
| `quartz-56` | generated | velvet | granular-cloud |
| `solstice-30` | generated | bright | fm-bleep |
| `orchid-56` | generated | close | pulse-train |
| `harbor-02` | generated | close | additive-pad |
| `ivory-18` | generated | close | fm-bleep |

All four designed presets and every generated preset pass the guard.

## CLI

| command | purpose |
| --- | --- |
| `vox generate --count N --seed S` | samples + presets + auditions + shortlist |
| `vox serve <run>` | shortlist with live tagging and auto-promotion |
| `vox promote <run> --kept a,b` | promote presets into `presets/shipped/` |
| `vox status` | shipped-set count against the 6–8 target |
| `vox report <run>` | table of a run's presets and guard numbers |
| `vox verify <run>` | prove the run rebuilds from its seeds |
| `vox render in.wav --preset p.json -o out.wav` | apply a preset to any audio |
| `vox phrase -o phrase.wav` | write the standard test phrase |
| `vox preset list \| validate` | inspect or validate preset files |

## Tests

```bash
.venv/bin/python -m pytest
```

78 tests: DSP primitives (STFT round-trip, WAV IO, resampling, effects), all
six sources (determinism, level, distinctness), preset schema and bounded
draws, guard behaviour (invariance to gain/delay, refusal of destructive
chains, WER monotonicity in noise), run generation and reproducibility,
promotion, and the tagging server over real HTTP.

## Layout

```
vox/
  dsp.py              WAV IO, STFT/ISTFT, resampling, effects
  synth.py            the six procedural sample sources
  color.py            pitch/formant/EQ/modulation render chain
  presets.py          schema, bounds, signatures, sampling
  testphrase.py       formant-synth standard test phrase
  intelligibility.py  STI-style estimator, WER mapping, optional ASR
  generate.py         runs, guard loop, artifacts, verification
  shortlist.py        shortlist HTML + tagging server
  promote.py          shipped-set promotion and index
  cli.py              command line
presets/
  designed/           the four hand-designed presets
  shipped/            curated generated presets + index.json
tests/
```

## License

MIT.
