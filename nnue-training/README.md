# Bermuda NNUE Experiments

This standalone workspace generates annotated self-play data and trains a
dual-perspective `Chess768 -> 64 -> 1` evaluator on CPU using NumPy. It does
not change the Bermuda engine build.

## Requirements

- Windows 10/11, PowerShell, Python 3.10 or later
- xmake and a release build of Bermuda
- Cutechess CLI (the workspace copy is at `..\cutechess-1.4.0-win64`)

Install the Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Generate Self-Play

Use the same Bermuda release binary, evaluation parameters, and hash setting
for both players. Start at concurrency one to keep CPU and memory use
predictable.

```powershell
$cutechess = "..\cutechess-1.4.0-win64\cutechess-cli.exe"
$engine = "..\build\windows\x64\release\bermuda.exe"
$openings = "..\experiments\eval-comparison-2026-11-01\openings.pgn"
New-Item -ItemType Directory -Force .\data | Out-Null

& $cutechess `
  -engine name=Bermuda-A cmd=$engine proto=uci option.Hash=32 `
  -engine name=Bermuda-B cmd=$engine proto=uci option.Hash=32 `
  -each tc=2+0.1 `
  -openings file=$openings format=pgn order=random plies=16 `
  -games 2 -rounds 10 -concurrency 1 -recover `
  -srand 20261104 `
  -pgnout .\data\pilot.pgn fi
```

This pilot plays 20 games. Check that the PGN contains eval comments such as
`{+0.57/8 0.22s}`, then convert the games to train and validation files:

```powershell
.\.venv\Scripts\python.exe .\generate_cutechess_dataset.py `
  --pgn .\data\pilot.pgn `
  --train-out .\data\train.txt `
  --validation-out .\data\validation.txt
```

Merge multiple completed PGNs when you want one archive. The glob is expanded
by the script, and source files are left untouched:

```powershell
.\.venv\Scripts\python.exe .\combine_pgns.py `
  --pgn .\data\pilot*.pgn `
  --out .\data\selfplay-combined.pgn
```

Each output row has Bullet's text-data shape:

```text
<FEN> | <white-relative score in centipawns> | <white-relative result>
```

The score is attached to the position before the move and converted from the
mover's perspective to White's perspective. The result is `1.0`, `0.5`, or
`0.0`. Mate, missing, shallow, and outlier scores are omitted, along with the
first 16 plies and positions where the mover is in check. Train and validation
are split by the position after the opening plies, keeping paired games on the
same side of the split. Both files are shuffled.

Tune filters with `--min-depth`, `--max-score-cp`, and `--min-ply`; set
`--split-ply` to match the opening depth, and use `--validation-percent` and
`--seed` for a reproducible split.

## Train

First verify the pipeline with a small subset and one epoch:

```powershell
.\.venv\Scripts\python.exe .\train_cpu.py `
  --data .\data\train.txt `
  --validation .\data\validation.txt `
  --max-positions 2000 --epochs 1 --batch-size 256 --threads 2
```

The trainer reports mixed-target, search-target, and WDL-target losses and
RMSE after each epoch, along with epoch duration, training positions/second,
and sampled peak RSS. It writes the same observation rows to
`checkpoints-cpu/training_observations.csv`, saves each epoch as a NumPy
`.npz`, and keeps `best.npz` selected by validation loss. Once the smoke test
works, use `--max-positions` only for debugging; omit it to use all rows.
For one million collected positions and a 10% holdout, five epochs are about
five passes over 900,000 training positions. Benchmark a smaller set first;
this simple CPU implementation prioritizes inspectability over throughput.

## Export and Engine Test

Export the checkpoint with the lowest validation loss. The current trainer
writes `best.npz`; earlier runs that predate this feature can export the
corresponding `epoch-NNN.npz` file instead. This engine format currently
requires a 64-neuron hidden layer.

```powershell
.\.venv\Scripts\python.exe .\export_nnue.py `
  --checkpoint .\checkpoints-cpu\best.npz `
  --output .\data\bermuda.nnue
```

In a UCI console, load the network and inspect the score:

```text
setoption name EvalFile value C:\path\to\bermuda.nnue
isready
position startpos
eval
go depth 6
```

Clear `EvalFile` to return to the handcrafted evaluator:

```text
setoption name EvalFile value
```

Search maintains per-ply incremental NNUE accumulators. The `eval` diagnostic
uses a full rebuild as a reference. Before interpreting match results, compare
exported-model scores against trainer inference on the same FENs.

## GitHub Self-Play Batches

The `Bermuda self-play data batch` workflow runs on a GitHub-hosted Windows
runner. Dispatch it with an even game count; it plays half as many opening
pairs, reversing colors, and uploads the PGN, Cutechess log, generated opening
suite, and run metadata as one artifact. Each run is bounded by the hosted-job
timeout and does not append data to the repository. Download each artifact and
merge its PGN locally with `combine_pgns.py` before dataset generation.

The default batch is 200 games at 5+0 with four concurrent games. Standard
GitHub runners provide four CPUs on public repositories; increase concurrency
only if you select a larger runner. An optional HTTPS `nnue_url` input lets a
run use a hosted `.nnue` file; with no URL the batch uses HCE. The workflow
file is on the `nnue-training` branch, but GitHub exposes manual
`workflow_dispatch` only after that workflow exists on the repository's
default branch. Merge the workflow there to enable the Actions UI button.

## Notes

The provided Bullet `ValueTrainerBuilder` configuration is GPU-oriented. Its
default `MockGpu` backend exists for type-checking and cannot execute training
kernels; it is not a CPU fallback. Bullet remains an option if a supported GPU
backend is available. The NumPy trainer here is the CPU path.

Keep the Bermuda binary, evaluation parameters, time control, and hash settings
fixed for each data version. Preserve source PGNs and record their generation
settings. Self-play scores are distillation targets from that teacher, not
stronger-than-teacher labels.
