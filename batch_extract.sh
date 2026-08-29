#!/usr/bin/env bash
#
# Run analyse.py over a folder of audio files and pull out one fixed set of
# descriptors.
#
#   ./batch_extract.sh recordings/
#   ./batch_extract.sh recordings/ -o dataset/ -s onset -j 8
#
# Writes, into the output directory:
#
#   full/<name>.segments.csv    the complete analysis for each file
#   selected/<name>.csv         just the descriptors below, per segment
#   all_segments.csv            every file's segments in one table
#   all_files.csv               one row per file (duration-weighted means)
#
# The full CSVs are kept because visualise.py and similarity.py read them;
# delete the full/ directory afterwards if you only want the extract.
#
# Re-running skips files whose analysis is already newer than the audio, so an
# interrupted batch can simply be started again.  Use --force to redo them.

set -euo pipefail

# --------------------------------------------------------------------------- #
# The descriptors to extract, as  full.column.name:output-name
#
# Three of the requested names existed in more than one group; the reading here
# is the spectral one for flatness and crest, and the psychoacoustic (sone) one
# for loudness -- the dynamics loudness is present too, as loudness_dB. Change
# a line here to change the extract; nothing else needs touching.
# --------------------------------------------------------------------------- #
DESCRIPTORS=(
  "dynamics.rms.mean:rms.mean"
  "spectral.centroid.mean:centroid.mean"
  "spectral.flatness.mean:flatness.mean"
  "spectral.flux.mean:flux.mean"
  "psycho.loudness.mean:loudness.mean"
  "psycho.sharpness.mean:sharpness.mean"
  "psycho.roughness.mean:roughness.mean"
  "dynamics.dynamic_complexity:dynamics.dynamic_complexity"
  "spectral.crest.mean:crest.mean"
  "spectral.complexity.mean:complexity.mean"
  "noisiness.dissonance.mean:dissonance.mean"
  "psycho.tonality.n_tones:tonality.n_tones"
  "psycho.tonality.strongest_tone_Hz:tonality.strongest_tone_Hz"
  "noisiness.f0.max:f0.max"
  "dynamics.loudness_dB:dynamics.loudness_dB"
)

# Only the groups and metrics the list above needs -- asking for less is the
# single biggest speed-up available, since the psychoacoustic models dominate.
# (Not named GROUPS: bash keeps a special variable of that name holding the
# current user's group ids, and assigning to it is silently ignored.)
ESSENTIA_GROUPS=(dynamics spectral noisiness)
PSYCHO_METRICS=(loudness sharpness roughness tonality)

EXTENSIONS=(wav aif aiff flac ogg mp3 m4a)

# --------------------------------------------------------------------------- #
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSE="$HERE/analyse.py"

INPUT=""
OUTDIR="analysis"
SEGMENTATION="fixed"
WINDOW=""
JOBS="1"
FORCE=0
EXTRA=()

usage() {
  cat >&2 <<USAGE
usage: $(basename "$0") FOLDER [options]

  -o DIR       output directory (default: analysis)
  -s METHOD    segmentation: fixed | onset | sbic | markers (default: fixed)
  -w SECONDS   window length for the fixed chop
  -j N         worker processes per file (default: 1)
  --force      re-analyse files that already have output
  --           everything after this is passed straight to analyse.py
  -h           this message

extracted descriptors:
$(printf '  %s\n' "${DESCRIPTORS[@]%%:*}")
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTDIR="$2"; shift 2 ;;
    -s) SEGMENTATION="$2"; shift 2 ;;
    -w) WINDOW="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *)
      if [[ -n "$INPUT" ]]; then echo "only one input folder, please" >&2; exit 2; fi
      INPUT="$1"; shift ;;
  esac
done

[[ -n "$INPUT" ]] || { usage; exit 2; }
[[ -d "$INPUT" ]] || { echo "not a folder: $INPUT" >&2; exit 2; }
[[ -f "$ANALYSE" ]] || { echo "cannot find analyse.py next to this script" >&2; exit 2; }

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || { echo "no $PYTHON on PATH" >&2; exit 2; }

FULL_DIR="$OUTDIR/full"
SEL_DIR="$OUTDIR/selected"
mkdir -p "$FULL_DIR" "$SEL_DIR"

FULL_NAMES=""; SHORT_NAMES=""
for pair in "${DESCRIPTORS[@]}"; do
  FULL_NAMES+="${pair%%:*} "
  SHORT_NAMES+="${pair#*:} "
done

# --------------------------------------------------------------------------- #
# Find the audio.  -print0 throughout, so spaces in names are not a problem.
# --------------------------------------------------------------------------- #
find_args=()
for ext in "${EXTENSIONS[@]}"; do
  find_args+=(-iname "*.${ext}" -o)
done
unset 'find_args[${#find_args[@]}-1]'   # drop the trailing -o

mapfile -d '' -t FILES < <(
  find "$INPUT" -maxdepth 1 -type f \( "${find_args[@]}" \) -print0 | sort -z
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "no audio files in $INPUT (looked for: ${EXTENSIONS[*]})" >&2
  exit 1
fi
echo "${#FILES[@]} file(s) in $INPUT" >&2

# --------------------------------------------------------------------------- #
# Analyse, then extract.
# --------------------------------------------------------------------------- #
analysed=0; skipped=0; failed=0
for audio in "${FILES[@]}"; do
  name="$(basename "$audio")"; stem="${name%.*}"
  segments="$FULL_DIR/${stem}.segments.csv"

  if [[ $FORCE -eq 0 && -s "$segments" && "$segments" -nt "$audio" ]]; then
    echo "  = $name (already analysed)" >&2
    skipped=$((skipped + 1))
  else
    echo "  → $name" >&2
    args=("$audio" --groups "${ESSENTIA_GROUPS[@]}"
          --psycho-metrics "${PSYCHO_METRICS[@]}"
          -s "$SEGMENTATION" -j "$JOBS" -o "$FULL_DIR" --no-labels -q)
    if [[ -n "$WINDOW" ]]; then args+=(--window "$WINDOW"); fi
    if [[ ${#EXTRA[@]} -gt 0 ]]; then args+=("${EXTRA[@]}"); fi
    if ! "$PYTHON" "$ANALYSE" "${args[@]}"; then
      echo "    failed, skipping" >&2
      failed=$((failed + 1))
      continue
    fi
    analysed=$((analysed + 1))
  fi

  [[ -s "$segments" ]] || { failed=$((failed + 1)); continue; }

  awk -v FULL="$FULL_NAMES" -v SHORT="$SHORT_NAMES" -v NAME="$stem" '
    BEGIN { FS = ","; OFS = ","; n = split(FULL, full, " "); split(SHORT, short, " ") }
    NR == 1 {
      for (i = 1; i <= NF; i++) col[$i] = i
      si = col["segment"]; ai = col["start"]; bi = col["end"]; di = col["duration"]
      header = "file,segment,start,end,duration"
      for (k = 1; k <= n; k++) {
        idx[k] = (full[k] in col) ? col[full[k]] : 0
        if (idx[k] == 0)
          printf("    warning: %s is not in %s\n", full[k], FILENAME) > "/dev/stderr"
        header = header "," short[k]
      }
      print header
      next
    }
    {
      row = NAME OFS $si OFS $ai OFS $bi OFS $di
      for (k = 1; k <= n; k++) row = row OFS (idx[k] ? $(idx[k]) : "")
      print row
    }
  ' "$segments" > "$SEL_DIR/${stem}.csv"
done

# --------------------------------------------------------------------------- #
# Combine: every segment of every file, and one row per file.
# --------------------------------------------------------------------------- #
combined="$OUTDIR/all_segments.csv"
per_file="$OUTDIR/all_files.csv"

first=1
: > "$combined"
for selected in "$SEL_DIR"/*.csv; do
  [[ -e "$selected" ]] || continue
  if [[ $first -eq 1 ]]; then cat "$selected" >> "$combined"; first=0
  else tail -n +2 "$selected" >> "$combined"; fi
done

# Duration-weighted, because segments differ in length under every
# segmentation but the fixed chop -- an unweighted mean would let a hundred
# short segments outvote the long one they sit inside.
awk -v SHORT="$SHORT_NAMES" '
  function numeric(v) { return v ~ /^-?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$/ }
  BEGIN { FS = ","; OFS = ","; n = split(SHORT, short, " ") }
  NR == 1 {
    for (i = 1; i <= NF; i++) col[$i] = i
    di = col["duration"]
    for (k = 1; k <= n; k++) idx[k] = col[short[k]]
    header = "file,n_segments,duration"
    for (k = 1; k <= n; k++) header = header "," short[k]
    print header
    next
  }
  {
    f = $1
    if (!(f in seen)) { seen[f] = 1; order[++count] = f }
    d = numeric($di) ? $di + 0 : 0
    segments[f]++; total[f] += d
    for (k = 1; k <= n; k++) {
      v = $(idx[k])
      if (numeric(v)) { sum[f, k] += v * d; weight[f, k] += d }
    }
  }
  END {
    for (i = 1; i <= count; i++) {
      f = order[i]
      row = f OFS segments[f] OFS sprintf("%.4f", total[f])
      for (k = 1; k <= n; k++)
        row = row OFS (weight[f, k] > 0 ? sprintf("%.6g", sum[f, k] / weight[f, k]) : "")
      print row
    }
  }
' "$combined" > "$per_file"

rows=$(($(wc -l < "$combined") - 1))
echo >&2
echo "analysed $analysed, reused $skipped, failed $failed" >&2
echo "wrote $SEL_DIR/*.csv" >&2
echo "wrote $combined  ($rows segment rows)" >&2
echo "wrote $per_file  ($(($(wc -l < "$per_file") - 1)) file rows)" >&2
