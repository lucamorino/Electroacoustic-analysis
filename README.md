# Electroacoustic-analysis

Descriptor extraction for long, texture-based electroacoustic pieces.

[Essentia](https://essentia.upf.edu/) provides the signal descriptors and the
onset detection; [MoSQITo](https://github.com/Eomys/MoSQITo) provides the
psychoacoustic ones (loudness in sone, sharpness in acum, roughness in asper,
tone-to-noise ratio in dB). How the audio gets sliced before any of that
happens is a swappable decision — a constant chop, detected onsets, timbral
change points, or your own markers.

Output is one CSV row per segment, one column per descriptor, plus a JSON
sidecar with the whole configuration and a piece-level summary, plus an
Audacity label track so you can audition the segmentation.

Three scripts, run in that order:

| script | does |
| --- | --- |
| `analyse.py` | audio → descriptors, one row per segment |
| `visualise.py` | that CSV → plots you can show and hide descriptors in |
| `similarity.py` | that CSV → which segments resemble each other, and the form that implies |

> **Status: draft.** The structure is in place and everything below runs, but
> the descriptor selection is a starting point for discussion rather than a
> settled analytical position. See *Where this is deliberately unfinished*.

## Install

```bash
pip install -r requirements.txt
```

`matplotlib` is in there because MoSQITo 1.2.x imports it at import time, not
because anything here plots. `PyYAML` is only needed for YAML config files —
JSON configs work without it.

## Quick start

```bash
# What is available
./analyse.py --list

# Audition a segmentation before committing to a long run — no descriptors,
# just the slicing and a label file you can drop into Audacity/Reaper
./analyse.py piece.wav --dry-run -s onset --onset-method complex

# Constant 5-second chop with 50% overlap, signal descriptors only (fast)
./analyse.py piece.wav -s fixed --window 5 --hop 2.5 --no-psycho -o analysis/

# One row per gesture, with the psychoacoustic set, on 8 cores
./analyse.py piece.wav -s onset --onset-method complex -j 8 -o analysis/

# A whole folder, driven by a config file
./analyse.py recordings/ --config configs/texture.yaml -o analysis/
```

`python -m eaa` is equivalent to `./analyse.py`. As a library:

```python
from eaa import AnalysisConfig, analyse

cfg = AnalysisConfig()
cfg.segmentation.method = "onset"
cfg.psychoacoustics.enabled = False
result = analyse("piece.wav", cfg)
print(result.rows[0]["spectral.centroid.mean"])
```

## How the audio is sliced

`-s/--segmentation` picks a strategy; the duration constraints
(`--min-duration`, `--max-duration`, `--pad`) then apply to all of them, so no
row ever hides three minutes of music and no row is 20 ms long.

| method | boundaries from | good for |
| --- | --- | --- |
| `fixed` | a constant grid (`--window`, `--hop`) | trajectories through the piece; anything you want to plot against time |
| `onset` | Essentia onset detection (`--onset-method`) | event- and gesture-based material |
| `sbic` | timbral change points (Bayesian Information Criterion over MFCCs) | sustained textures, where the spectrum changes but nothing resembles an attack |
| `markers` | a label file you wrote (`--markers`) | testing an analysis against your own formal reading of the piece |

For `onset`, `complex` responds to amplitude *and* phase change and so catches
soft spectral entries that an energy-based detector misses; `hfc` favours
percussive attacks; `superflux` is designed to ignore vibrato. `--onset-alpha`
is the peak-picking threshold — lower it for more onsets.

`sbic` is the one worth knowing about if the material is sustained: it looks
for points where a sequence of feature vectors is better described by two
Gaussians than by one. Its `sbic_cpw` penalty (config file only) controls
sensitivity — the default 1.5 is conservative, 0.5 finds considerably more.

Adding a strategy means writing a function that returns `(start, end)` pairs
and decorating it with `@register("name")` in `eaa/segmentation.py`.

## What gets measured

Essentia descriptors are computed on inner frames (`--frame-size`,
`--hop-size`) *within* each segment, then aggregated. So every column is a pair
of what was measured and how it was summarised: `spectral.centroid.mean` and
`spectral.centroid.stdev` say different and equally interesting things about a
texture. Statistics come from Essentia's `PoolAggregator` (`--stats`).

Groups are selected with `--groups`; the default is `dynamics spectral
noisiness mfcc barkbands`.

| group | descriptors |
| --- | --- |
| `dynamics` | RMS, Stevens loudness, power in dB, dynamic complexity |
| `spectral` | centroid, spread, skewness, kurtosis, rolloff, decrease, flatness, crest, entropy, HFC, complexity, strong peak, flux, energy band ratios |
| `noisiness` | ZCR, pitch salience, YIN f0 + confidence, peak count, dissonance |
| `mfcc` / `gfcc` | cepstral coefficients |
| `barkbands` / `melbands` / `erbbands` | band energies plus their flatness and crest |
| `harmonic` | inharmonicity, tristimulus, odd-to-even ratio |
| `tonal` | HPCP profile, its entropy and crest |

Psychoacoustic metrics (`--psycho-metrics`, default `loudness sharpness
roughness tonality`) report mean, max, std and exceedance percentiles — `N5` is
the loudness exceeded during 5% of the segment, which for a fluctuating texture
is usually more telling than the mean. `loudness` also reports the centre of
gravity of the specific-loudness pattern in bark, as a perceptual counterpart
to the spectral centroid. `loudness_ecma` and `roughness_ecma` (ECMA-418-2) are
implemented but off by default because they are slow.

Whole-file descriptors — EBU R128 integrated loudness and loudness range,
dynamic complexity — go into the JSON under `metadata.global`.

### Calibration

MoSQITo's models expect a signal in pascals, so they need to know what level
full scale represents. `--spl-full-scale` (default 94 dB, i.e. a full-scale
sine reads 1 Pa RMS) is that reference. Set it to match your monitoring chain
if you want absolute numbers; leave it alone and read the values comparatively
within the piece. `--normalize` invalidates this, which is why it is off.

## Performance

Measured on 4 cores, on a 40-second file:

| configuration | speed |
| --- | --- |
| Essentia descriptors only | ~19× real time |
| plus the four default psychoacoustic metrics, `-j 4` | ~1.3× real time |

The psychoacoustic models dominate — roughly 3 s of compute per 2 s of audio,
per core. That is fine for a 10-minute piece and painful for a 60-minute one,
so: use `--dry-run` to settle the segmentation first, `-j` to fill your cores,
`--psycho-max-duration` to cap what the slow metrics see of each segment, and
`--no-psycho` while you are still deciding what you want. Loudness and
sharpness share one underlying computation, so asking for both costs barely
more than asking for one.

## Looking at the result

```bash
# A mixed overview: level, spectral shape, noisiness, perception
./visualise.py analysis/piece.segments.csv

# Interactive page: every descriptor as a checkbox, shared crosshair, table view
./visualise.py analysis/piece.segments.csv --html

# Pick your own, with globs; compare shapes on one z-scored axis
./visualise.py analysis/piece.segments.csv -d 'psycho.*.mean' --overlay
./visualise.py analysis/piece.segments.csv -d 'spectral.*.mean' --png shape.png

# What is in this CSV?
./visualise.py analysis/piece.segments.csv --list
```

Descriptors are drawn as **small multiples** — one panel each, own y-axis,
shared time axis. That is deliberate rather than lazy: a spectral centroid in
Hz and a roughness in asper share no scale, and putting them on one pair of
axes invents a relationship the data does not contain. `--overlay` does put
them on one axis, but only after z-scoring each descriptor, so the shared axis
honestly means "standard deviations from this descriptor's own mean" — that is
the view for asking which descriptors move together.

Values are drawn as **steps**, not as a line through segment midpoints: a
segment's value describes an extent of time, and a sloped line between
midpoints would imply a smoothness nothing measured.

Show/hide works in both outputs. The matplotlib window has checkboxes down the
left side; hidden panels give their vertical space back to the rest. `--html`
writes one self-contained file — no CDN, no server, so it survives being
emailed — where every descriptor in the CSV is a checkbox grouped by family, a
shared crosshair reports every visible descriptor at one instant, and a table
view gives the same numbers without hovering. `--smooth N` takes a rolling
median, which is what makes an onset-sliced run of several hundred short
segments readable.

## Which segments resemble each other

```bash
# Cluster on timbral descriptors, number of clusters chosen by silhouette
./similarity.py analysis/piece.segments.csv

# Fix the number of clusters; try a different notion of distance
./similarity.py analysis/piece.segments.csv -k 5 --metric cosine

# Compare on your own features
./similarity.py analysis/piece.segments.csv -f 'mfcc.*' 'psycho.*' --method kmeans
```

The default feature set is deliberately **timbral, and excludes level and
duration**, so the same gesture played loudly and quietly still counts as a
resemblance. Descriptors are standardised before anything is compared —
otherwise a centroid in Hz would outvote a flatness in 0..1 by four orders of
magnitude.

It prints a report and writes four plots plus three files:

- **`.ssm.png`** — the self-similarity matrix. Where the piece returns to
  itself, as blocks off the diagonal. Usually the single most informative
  picture of a long piece.
- **`.dendrogram.png`** — the hierarchy of resemblances, with the cut that
  produced the clusters marked.
- **`.timeline.png`** — the clusters laid out in time (the form, as the
  features see it), with each segment's silhouette underneath so you can see
  where the grouping is confident and where it is guessing.
- **`.clusters.png`** — the segments in the plane of the two leading principal
  components, faceted one panel per cluster when there are more than three.
- **`.clusters.csv`**, **`.neighbours.csv`** — cluster and silhouette per
  segment; the closest few segments to every segment.
- **`.clusters.labels.txt`** — an Audacity label track named by cluster. This
  is the one to keep: load it beside the audio and the form is annotated, so
  you can listen to what the clustering actually claims.

Two things the report tells you that matter more than the labels:

**The mean silhouette.** It is printed for every candidate k. Near 1 the
clusters are genuinely separated; near 0 they are not, and the labels are an
arbitrary partition of a continuum — which for a slowly-evolving texture is a
perfectly real answer about the piece, not a failure.

**What each cluster is.** Because the features are standardised over the whole
piece, a cluster's mean on a feature is directly "how many standard deviations
this group sits from the piece average". The report lists the largest of those
per cluster, which is what turns `c3` into "low-band energy, low centroid — the
drone".

## Output

For `piece.wav` in `-o analysis/`:

- `piece.segments.csv` — one row per segment; leading columns are `segment`,
  `start`, `end`, `duration`, all in seconds and absolute in the source file.
- `piece.analysis.json` — the same rows, plus `metadata` (including the
  whole-file descriptors), the full `config` used, and a duration-weighted
  `summary` over the piece. The weighting matters under onset segmentation,
  where a hundred 40 ms clicks would otherwise outvote the drone they sit on.
- `piece.labels.txt` — Audacity label track of the segmentation.

## Layout

```
analyse.py                    audio -> descriptors (== python -m eaa)
visualise.py                  CSV -> plots, static or interactive
similarity.py                 CSV -> resemblance, clusters, form
configs/                      example configs: texture.yaml, gesture.yaml, fast.json

eaa/config.py                 every knob, as dataclasses
eaa/audio.py                  loading, resampling, SPL calibration
eaa/segmentation.py           the slicing strategies + shared constraints
eaa/descriptors_essentia.py   Essentia descriptor groups
eaa/descriptors_psycho.py     MoSQITo metrics
eaa/pipeline.py               orchestration, parallelism, summary
eaa/export.py                 CSV / JSON / labels
eaa/cli.py                    argument parsing

eaa/table.py                  reading the CSV back, resolving descriptor globs
eaa/plotstyle.py              one palette and set of mark rules for every plot
eaa/viz_mpl.py                descriptor panels (matplotlib)
eaa/viz_html.py               the self-contained interactive page
eaa/similarity.py             distances, clustering, silhouette, PCA
eaa/viz_similarity.py         SSM, dendrogram, timeline, cluster gallery
```

Config precedence is defaults ← config file ← command-line flags.

## Where this is deliberately unfinished

- **No novelty-based structure detection.** The self-similarity matrix is
  there, but not a Foote-style checkerboard novelty curve over it, which would
  find boundaries rather than assume the segmentation's.
- **Clustering assumes the segmentation.** If the slicing is wrong, the
  clusters inherit that. The SSM is the honest thing to read first.
- **Descriptor choice is provisional.** `harmonic` and `tonal` in particular
  assume a pitch that a lot of this repertoire does not have; they are included
  for completeness, not because they are recommended.
- **A failing descriptor warns once and yields an absent column** rather than
  stopping the run. Good for long batch jobs, so check the warnings.
- **Stereo is downmixed** for analysis. Spatial descriptors (width, correlation,
  per-channel differences) would be a real addition for this repertoire.
