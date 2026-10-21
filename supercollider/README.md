# SCMIR analysis in SuperCollider

Descriptor extraction for one audio file using Nick Collins's
[SCMIR](https://github.com/sicklincoln/SCMIR) library, following the method of

> Nick Collins, "Noise Music Information Retrieval", in *Noise in and as Music*
> (University of Huddersfield Press, 2013), doi:[10.5920/noise.05](https://doi.org/10.5920/noise.05)

This is a separate route from the Python pipeline in the rest of the
repository, but it writes CSV files that `visualise.py` and `similarity.py` read
directly (see [Using the results](#using-the-results)).

| file | what it is |
| --- | --- |
| `scmir_analysis.scd` | the analysis: settings at the top, run it on a file |
| `scmir_check_install.scd` | checks the installation without analysing anything |

## What it computes

The defaults reproduce the paper's set-up.

**Six features** — the ones Collins chose as most relevant to noisy, textural
material:

| SCMIR feature | measures | nearest descriptor in the Python pipeline |
| --- | --- | --- |
| `Loudness` | perceptual loudness, in sones | `psycho.loudness` (a different model, MoSQITo's Zwicker) |
| `Transient` | "transientness": wavelet transient energy | none directly |
| `SensoryDissonance` | sensory dissonance | `noisiness.dissonance` |
| `FFTCrest` | spectral crest: how peaky the spectrum is | `spectral.crest` |
| `SpecCentroid` | spectral centroid, Hz | `spectral.centroid` |
| `SpectralEntropy` | spectral entropy | `spectral.entropy` |

The third column names the closest counterpart, not the same algorithm. The
implementations differ, so compare shapes across the two pipelines, never raw
values.

**Per segment**, the mean and maximum of each feature. By default segments are
one-second windows, as in the paper. They can instead run from onset to onset,
or be every analysis frame (about 23 ms).

**Over the whole file**, the mean and maximum of each feature. The paper calls
these the summary features.

**Sections:** SCMIR's `findSections`, a novelty curve over the self-similarity
matrix of the features, peak-picked into boundaries. In the paper it found the
92-second turn in Merzbow's *Silent Night*. Read the caveat under
[What testing showed](#what-testing-showed) before trusting the exact times.

**Normalised copy:** the per-segment values are also written min–max
normalised to 0..1 within the file, the scale the paper's figures use.

### What is not here

The paper goes on to compare *many* files. It normalises globally across a
corpus, quantises one-second feature vectors with k-means (20 clusters), and
trains variable-order Markov models to get cross-likelihood similarity and
"dominance" matrices. That is a corpus-level step and is not implemented. SCMIR
has the pieces (`SCMIR.findGlobalFeatureNorms`, `PPMC`), so it is a natural
next step once single-file runs are settled.

## Installing

You need SuperCollider (3.13 is what SCMIR 1.6 is tested on), three add-ons,
and optionally a fourth. Every add-on goes into your **Extensions folder**. To
find it, evaluate this in SuperCollider:

```supercollider
Platform.userExtensionDir
```

On macOS it is `~/Library/Application Support/SuperCollider/Extensions`.

### 1. sc3-plugins

These provide the UGens SCMIR needs: `FeatureSave` (which SCMIR uses for every
extraction), `SensoryDissonance`, `SpectralEntropy` and `FFTCrest`.

1. Download the release matching your SuperCollider version from
   <https://github.com/supercollider/sc3-plugins/releases>.
2. Unzip it, and move the `SC3plugins` folder into the Extensions folder.
3. **macOS only:** downloaded plugins are quarantined and will not load until
   you clear the flag. In Terminal:

   ```sh
   xattr -rd com.apple.quarantine ~/Library/Application\ Support/SuperCollider/Extensions/SC3plugins
   ```

### 2. SCMIR

1. Download <https://github.com/sicklincoln/SCMIR> (Code ▸ Download ZIP, or `git clone`).
2. Copy **only the `SCMIRExtensions` folder** into the Extensions folder.

Don't install the whole repository, and don't install it as a Quark. Its
`AdditionalExamples` folder contains an extension to `KMeans`, a class from a
separate Quark. Without that Quark installed, the class library fails to
compile.

### 3. Helper programs, for sections

`findSections` hands the similarity matrix and novelty curve to two small
compiled programs, `similaritymatrix` and `noveltycurve`. They live in a folder
called `scmirexec` inside `SCMIRExtensions`:

```
Extensions/
  SCMIRExtensions/
    Classes/
    scmirexec/
      similaritymatrix
      noveltycurve
```

Without them, everything else still runs; the script skips sections and says so.

There are two ways to get them:

- **Prebuilt (macOS).** SCMIR's own install notes point to
  `composerprogrammer.com/code/scmirexec.zip`, with builds for Intel and Apple
  Silicon. Put the ones for your machine in `scmirexec/`, then clear the
  quarantine flag and make them executable:

  ```sh
  cd ~/Library/Application\ Support/SuperCollider/Extensions/SCMIRExtensions/scmirexec
  xattr -d com.apple.quarantine similaritymatrix noveltycurve
  chmod +x similaritymatrix noveltycurve
  ```

- **Compile them yourself.** This takes seconds and needs no libraries. On
  macOS, run `xcode-select --install` first if you've never compiled anything.
  From inside the downloaded SCMIR folder:

  ```sh
  EXEC=~/Library/Application\ Support/SuperCollider/Extensions/SCMIRExtensions/scmirexec
  mkdir -p "$EXEC"
  c++ -O2 -o "$EXEC/similaritymatrix" Source/similaritymatrix/similarity.cpp
  c++ -O2 -o "$EXEC/noveltycurve"     Source/noveltycurve/noveltycurve.cpp
  ```

  (On Linux, replace the path with `~/.local/share/SuperCollider/Extensions/...`.)

### 4. Wavelet UGens, for `Transient` (optional)

`Transient` needs `DWT`, `WT_Transient` and `WT_ModulusSum`. These are Nick
Collins's wavelet UGens and are **not part of sc3-plugins**. He distributes
them from his code page, composerprogrammer.com/code.html. I couldn't reach
that site to confirm the exact package, so check there.

Without them, the script drops `Transient`, says so, and analyses with the other
five features.

### 5. Recompile and check

Recompile the class library: **Language ▸ Recompile Class Library**, or
Cmd+Shift+L. Then open `scmir_check_install.scd`, click inside its block and
press Cmd+Return. The Post window reports what is installed, what each feature
needs, and whether sections are available:

```
Installation OK.
  SuperCollider 3.13.0
  Extensions folder: /Users/you/Library/Application Support/SuperCollider/Extensions
  Features available: [ Loudness, SensoryDissonance, FFTCrest, SpecCentroid, SpectralEntropy ]
  Sections (findSections): available
  Target: ~/Music/piece.wav - not found (set `target` before analysing)
```

## Running

**In the IDE.** Open `scmir_analysis.scd` and set `target` at the top of the
settings. Then click anywhere inside the block that starts on the `(` line and
press Cmd+Return (Ctrl+Return on Linux or Windows). Progress appears in the Post
window:

```
[1/5] Reading /Users/you/Music/piece.wav
[2/5] Extracting [ Loudness, SensoryDissonance, FFTCrest, SpecCentroid, SpectralEntropy ] ...
[3/5] Segmenting: fixed
[4/5] Sections
      boundaries at [ 10.22, 22.29, 35.29, 45.51 ] s
[5/5] Writing
```

It ends with a table of whole-file means and maxima and the list of files
written.

**From a terminal.** The path given on the command line replaces `target`, and
an optional second path sets the output folder:

```sh
/Applications/SuperCollider.app/Contents/MacOS/sclang scmir_analysis.scd ~/Music/piece.wav
/Applications/SuperCollider.app/Contents/MacOS/sclang scmir_analysis.scd ~/Music/piece.wav ~/Desktop/results
```

From a terminal, sclang quits when the run finishes, with status 0 for success
and 1 for an error, so it can be looped over a folder.

Speed is not an issue: a 60-second file takes under a second, and a long piece
takes seconds to a minute.

### Settings

Everything is at the top of `scmir_analysis.scd`.

| setting | default | |
| --- | --- | --- |
| `target` | `"~/Music/piece.wav"` | the audio file |
| `start`, `duration` | `0`, `0` | analyse an excerpt, in seconds (`duration` 0 = to the end) |
| `outputDir` | `nil` | `nil` = a folder `scmir` beside the audio |
| `features` | the paper's six | any of the features listed at the bottom of the script |
| `skipMissingFeatures` | `true` | drop a feature whose UGens are missing, or stop |
| `frameHop` | `1024` | analysis hop in samples: 1024 (~23 ms) or 512 (~12 ms), nothing else |
| `segmentation` | `\fixed` | `\fixed`, `\onsets` or `\frames` |
| `window`, `hop` | `1.0`, `nil` | `\fixed` window and step in seconds; `nil` hop = no overlap |
| `onsetThreshold`, `onsetODF` | `0.5`, `\power` | `\onsets` detector; see below on `\rcomplex` |
| `onsetMinGap`, `minSegment` | `10`, `0.05` | minimum gap (512-sample frames) and minimum segment (s) |
| `stats` | `[\mean, \max]` | any of `\mean \max \min \stdev \median` |
| `findSections` | `true` | needs the helper programs |
| `sectionMetric` | `0` | 0 cosine, 1 Manhattan, 2 Euclidean |
| `sectionUnit`, `sectionKernel` | `10`, `43` | cell size in frames; window in cells (10 × 43 ≈ 10 s) |
| `writeNormalised` | `true` | also write the 0..1 per-file normalised table |
| `writeReaper` | `false` | also write REAPER region CSVs beside the Audacity labels |

Features take optional parameters after the name, e.g.
`[\SensoryDissonance, 100, 0.1, nil, 10]`. The bottom of the script lists every
feature SCMIR supports, what each needs installed, and the parameters for the
paper's six.

## Outputs

For `piece.wav`, in `scmir/` beside it (an excerpt adds the range to the name,
e.g. `piece_24.0-44.0s`):

| file | contents |
| --- | --- |
| `piece.scmir.segments.csv` | one row per segment: `segment, start, end, duration`, then `scmir.<Feature>.<stat>` for each feature and statistic; raw values |
| `piece.scmir.segments.normalised.csv` | the same, min–max normalised to 0..1 within this file |
| `piece.scmir.frames.csv` | instead of the two above, with `segmentation: \frames`; one row per analysis frame |
| `piece.scmir.summary.csv` | one row: the whole-file statistics, raw values |
| `piece.scmir.sections.csv` | the sections between `findSections` boundaries: `section, start, end, duration` |
| `piece.scmir.sections.labels.txt` | the same as an Audacity label track (File ▸ Import ▸ Labels…) |
| `piece.scmir.onsets.labels.txt` | with `\onsets`: one label per onset segment |
| `*.reaper.csv` | with `writeReaper: true`: the label tracks as REAPER regions (Region/Marker Manager ▸ Import…) |
| `piece.scmir.run.json` | provenance: every setting, the audio file's details, SuperCollider version, features computed, warnings, outputs |

Times are always absolute in the source file, including for excerpts.

The raw and normalised tables answer different questions. Raw values compare
across files. Normalised values show where each moment sits within this
piece's own range: comparable in shape, not in level. The paper normalises
across a whole corpus at once, which is the corpus step described under
[What is not here](#what-is-not-here).

## Using the results

The segment and frame tables have the same leading columns as `analyse.py`'s
output, so the repository's Python tools read them as they are. Name the SCMIR
columns explicitly, because the tools' default descriptor sets look for the
Python pipeline's names:

```sh
./visualise.py ~/Music/scmir/piece.scmir.segments.csv -d 'scmir.*.mean'
./visualise.py ~/Music/scmir/piece.scmir.segments.csv -d 'scmir.*' --html
./similarity.py ~/Music/scmir/piece.scmir.segments.csv -f 'scmir.*.mean'
```

## What testing showed

I ran the script on SuperCollider 3.13 and sc3-plugins under Linux, with SCMIR
installed exactly as above. The test piece was 60 seconds of synthetic audio
with a known form: a noise bed, a low drone, the noise again, a section of
sparse high pings, and the noise again, 12 seconds each. The features step
cleanly at 12, 24, 36 and 48 s. These are the things worth knowing before
reading results from real pieces.

**Section boundaries come out early.** `findSections` found all four
boundaries, but between 0.7 and 2.5 s early (10.2, 22.3, 35.3, 45.5 s). The
features themselves change exactly on the boundary, so the bias comes from
SCMIR's novelty step. In `Source/noveltycurve/noveltycurve.cpp`, the
checkerboard sign is computed but never applied (`sum += matrix[...]`, not
`mult * matrix[...]`). The novelty curve therefore measures how mixed a whole
window is, rather than Foote's contrast across its centre, and its peaks lean
towards the busier side of a boundary. The bias scales with the window.
Halving it (`sectionUnit: 5`, default kernel) halved the mean error, from 1.7 s
to 0.8 s, with the worst case at 1.3 s. The script uses SCMIR's code as is, since that is what the paper
used. Treat the times as approximate and check them by ear against the label
track.

**`SensoryDissonance` saturates on noise.** The UGen clamps its output at 1.0
by default, and the noise bed sat at the clamp in 21% of frames. Raising the
clamp to 10 showed a true peak of 2.1 and a mean 24% higher. Collins reports
the same with Merzbow, where he had to reset the threshold. The script warns
whenever more than 1% of frames sit at the clamp, and tells you how to raise
it. Only compare runs that use the same clamp.

**SCMIR's default onset detector over-triggers on drones.** With
`\rcomplex`, SCMIR's default, the 12-second drone produced 113 false onsets.
Its 512-sample frames are shorter than one period of the drone's 70 Hz
fundamental, so the phase prediction never settles. `\power` gave none in the
drone and caught all nine pings within 6 ms, so it is the script's default.
Drone-based repertoire will hit this.

**48 kHz files work, with a small offset.** SCMIR analyses at 44.1 kHz and
resamples other rates on the fly. Timing and pitch are unaffected: a 220 Hz
tone reads the same at either rate. Broadband descriptors read a few per cent
lower from a 48 kHz file: white noise's centroid was 3.6% lower. Keep a
collection you intend to compare at one sample rate, ideally 44.1 kHz.

**Four SCMIR quirks the script works around:**

- SCMIR only posts a message if the audio file can't be opened, then fails later
  with an unrelated error. The script opens the file first.
- `SCMIRAudioFile` takes its start as a *fraction* of the file but its duration
  in seconds. With a start set, a duration of 0 reads past the end of the file.
  The script converts and passes both correctly.
- Cosine distance between single-number vectors is always 0, so with one
  feature `findSections` finds nothing. The script switches to Euclidean and
  says so.
- SCMIR's per-segment maximum starts counting from 0. For features that can be
  negative (MFCCs, for example), a segment that is all-negative reports a max
  of 0. The script warns when a chosen feature has negative values.

## Troubleshooting

| message | cause |
| --- | --- |
| `SCMIR is not installed` | `SCMIRExtensions` not in the Extensions folder, or the class library not recompiled |
| `FeatureSave is missing` | sc3-plugins not installed, or (macOS) still quarantined |
| `\Transient needs DWT, ...` | the wavelet UGens aren't installed; the run continues with the other features |
| `feature extraction returned no data` | a server plugin is missing although its class exists. On macOS this is usually the quarantine flag on sc3-plugins |
| `findSections needs SCMIR's helper programs` | `scmirexec/` missing or not executable; see [Helper programs](#3-helper-programs-for-sections) |
| `the file is 96000 Hz` | convert to 44.1 kHz first |
| class library fails to compile, mentioning `KMeans` | the whole SCMIR repository was installed; keep only `SCMIRExtensions` |
