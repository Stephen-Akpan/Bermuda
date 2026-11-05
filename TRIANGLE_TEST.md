# TriangleTest

TriangleTest is a small, file-backed evaluation-and-strength workflow for
Bermuda. It separates an experiment into three stages and records each stage's
inputs and outputs so a run can be inspected, stopped, and resumed.

1. **Data preparation:** PGN games are parsed, legal moves are validated, and
   each pre-move position is written to a game-level training or validation
   dataset.
2. **Parameter optimization:** Bermuda's Texel/SPSA tuner starts from the
   selected current weights, evaluates the held-out split, writes the best
   candidate, and checkpoints its optimizer state every ten iterations.
3. **Strength evaluation:** Cutechess plays paired games with colors reversed
   for each reproducibly randomized opening. It runs several pairs per process
   to reduce startup overhead. Completed opening pairs are recorded in a
   manifest and can be skipped on resume. The combined PGN is retained for
   further analysis with Ordo or other tools.

## Requirements

* Windows PowerShell.
* Python 3 (standard library only) for paired-result validation and SPRT
  calculation. The executable defaults to `python`; use `-PythonPath` to select
  another Python 3 installation.
* Build `bermuda_tuner` and `bermuda` with xmake.
* Cutechess CLI. By default TriangleTest uses the copy at
  `cutechess-1.4.0-win64\cutechess-cli.exe`.
* A completed-game PGN dataset, a current Bermuda parameter file, and a PGN
  opening suite.

```powershell
xmake build bermuda
xmake build bermuda_tuner
```

## Run a complete experiment

Use a unique `RunId` for each candidate experiment. The example uses a current
parameter file from the release build and the repository's existing sample
PGNs; substitute your own paths and test settings.

```powershell
.\tools\TriangleTest.ps1 `
  -Stage All `
  -RunId hce-2026-10-27-a `
  -Pgn .\build\windows\x64\release\Dataset.pgn `
  -CurrentParams .\build\windows\x64\release\eval_params.txt `
  -Openings .\build\windows\x64\release\openings.pgn `
  -ValidationPercent 10 `
  -Seed 12345 `
  -Family all `
  -Iterations 1000 `
  -Batch 4096 `
  -Games 10000 `
  -TimeControl '30+0' `
  -PairsPerBatch 10 `
  -HashMb 64 `
  -MoveOverhead 10 `
  -Concurrency 1 `
  -OpeningPlies 16
```

Run outputs are stored under `experiments\TriangleTest\<RunId>\` by default.
`-RunRoot` can point that directory elsewhere. The run directory contains:

| File or directory | Purpose |
|---|---|
| `triangle-manifest.json` | Run identity, input/pair hashes, options, SPRT state, completed match pairs |
| `train.data`, `validation.data` | Validated `GAME_ID RESULT FEN` rows |
| `optimizer.checkpoint` | Current and best weights, iteration, RNG state, options, and dataset fingerprint |
| `optimizer.checkpoint.bak` | Previous checkpoint, used if the latest checkpoint is invalid |
| `candidate-params.txt` | Best held-out parameter set found so far |
| `matches\pair-*.pgn` | One paired opening encounter per file |
| `matches\pair-*.log` | Cutechess output for each encounter |
| `matches\batch-*.pgn` | Raw Cutechess output for a batch before pair splitting |
| `paired-match.pgn` | Combined completed match games, ready for independent analysis |
| `promotion-report.txt` | Paired W/D/L summary and sequential SPRT decision |
| `reproducibility.json` | Hardware, runtime, binary/input hashes, UCI settings, and test policy |
| `promoted-params.txt` | Separate copy made only after the promotion gate passes and `-Promote` is supplied |

The script pins the input file hashes and relevant options in the manifest.
Changing the PGN, split settings, initial weights, tuning parameters, match
baseline, openings, time control, concurrency, or opening depth requires a new
`RunId`. This prevents a resumed experiment from silently mixing conditions.
Match options also pin batch size, Hash, MoveOverhead, machine/runtime details,
and analyzer version.
Run manifests created before the SPRT schema also require a new `RunId` for
strength testing.

## Running and resuming one stage

Stages can be run separately:

```powershell
.\tools\TriangleTest.ps1 -Stage Prepare -RunId hce-2026-10-27-a `
  -Pgn .\games.pgn -Seed 12345 -ValidationPercent 10

.\tools\TriangleTest.ps1 -Stage Tune -RunId hce-2026-10-27-a `
  -CurrentParams .\eval_params.txt -Seed 12345 -Iterations 1000 -Batch 4096

.\tools\TriangleTest.ps1 -Stage Match -RunId hce-2026-10-27-a `
  -CurrentParams .\eval_params.txt -Openings .\openings.pgn `
  -Games 10000 -TimeControl '30+0' -PairsPerBatch 10

# Identical-baseline control; use a fresh RunId for each independent run.
.\tools\TriangleTest.ps1 -Stage Calibrate -RunId control-2026-10-27-a `
  -CurrentParams .\eval_params.txt -Openings .\openings.pgn `
  -Games 10000 -TimeControl '30+0' -PairsPerBatch 10
```

Rerun the same stage with the same `RunId` and options to resume. `-Stage
Status` displays the manifest:

```powershell
.\tools\TriangleTest.ps1 -Stage Status -RunId hce-2026-10-27-a
```

### Stop and resume behavior

* **Preparation** writes to temporary files before finalizing the split; once
  the manifest marks it complete, the source PGN and generated datasets are
  hashed and held fixed.
* **Optimization** checkpoints every ten completed SPSA iterations and saves
  the best candidate at the same time. If interrupted between checkpoints,
  at most nine iterations are repeated. Resume with identical optimization
  options; the tuner verifies both the dataset fingerprint and checkpoint
  settings and restores the SPSA random-number generator state.
* **Strength testing** runs several paired opening encounters per Cutechess
  process (`-PairsPerBatch`, default 10). The manifest advances only after a
  whole batch completes and each two-game pair is validated. If stopped
  during a batch, that batch is retried; earlier completed pairs remain
  intact. A failed batch PGN is archived and excluded. After a successful
  batch, TriangleTest updates the SPRT and manifest pair-by-pair. If the SPRT
  crosses a boundary partway through a batch, later played pairs are kept as
  `unused-after-stop-*` and excluded from the result. During an uninterrupted
  run only the new pair is analyzed using the preceding SPRT state. On resume,
  the analyzer recomputes counts and overshoot history from all
  manifest-recorded pairs instead of trusting cached LLR state.

Cutechess's `-recover` option is enabled to recover from engine crashes within
an encounter. TriangleTest's manifest, rather than `-recover`, handles
resuming between encounters.

## Data preparation details

`bermuda_tuner --prepare-pgn` uses the tuner PGN parser. It validates PGN
structure, result tags, FEN setup positions, SAN, and move legality while
walking each completed game. The dataset contains the position before every
played move and labels those positions with the final game result from White's
perspective. Unfinished games with no decisive or drawn result are omitted.
Invalid or ambiguous moves fail preparation instead of silently emitting a
partial dataset.

The split is shuffled deterministically by game using `Seed`, never by
individual position. As a result, positions from a game cannot leak between
training and validation. Both splits must contain at least two completed
games. The tuner also checks each FEN and rejects rows where the same game ID
has conflicting results.

Game outcomes are noisy labels for individual positions: a position is not a
direct engine evaluation target just because its game was won or lost. Use
large, representative PGNs, avoid excessive correlated positions from the
same games, and reserve an untouched external dataset for final checks.

## Parameter optimization details

The workflow tunes the current supported 19 evaluation scales through
SPSA/Texel logistic loss. The validation split is used to select the best
candidate; it is not used to estimate playing Elo. `--family` can restrict
optimization to `all`, `material`, or `piece-values`.

The optimizer checkpoint contains the current and best parameter vectors,
completed iteration count, validation/training losses, RNG state, data
fingerprint, and optimization options. A checkpoint cannot be resumed against
different datasets, family, iteration target, batch size, or logistic K. Keep
the target iteration count constant for a run so the SPSA learning-rate
schedule remains the same.

## Strength evaluation and promotion

Each Cutechess encounter uses one opening from the supplied PGN suite, with
the same opening played once by each engine color. Test the candidate against
the baseline parameter file using identical Bermuda binaries, hash settings,
and clock controls. TriangleTest sets Hash and MoveOverhead explicitly for
both engines. Bermuda does not currently expose a UCI Threads option.
`Concurrency` defaults to 1 intentionally: raise it only after checking that
concurrent games do not change time-loss rates or contaminate throughput.

TriangleTest stops when an SPRT boundary is crossed or the `-Games` maximum is
reached, combining completed PGNs and updating a Fishtest-style logistic SPRT
after each pair. The two games from one opening (one with each engine color)
form a single pentanomial observation:
LL, LD+DL, LW+DD+WL, DW+WD, or WW. This uses the paired design instead of
counting the two same-opening games as independent results. Candidate must
switch colors and both games must be complete for the pair to be accepted.

The initial policy is alpha=0.05, beta=0.05, H0=0 Elo, and H1=+5 Elo. The
corresponding LLR boundaries are approximately -2.944 and +2.944. Crossing
the upper boundary reports **PASS** (evidence for H1); crossing the lower
boundary reports **FAIL** (evidence against H1). A PASS is not a claim that the
candidate is exactly +5 Elo, and a FAIL does not prove it is exactly 0 Elo.
Reaching the `-Games` maximum before either boundary reports
**INCONCLUSIVE**. The default maximum is 10,000 games; this is a cap, not a
guarantee that the test will decide. TriangleTest promotes only a PASS when
`-Promote` is supplied, and never replaces the baseline or current parameter
file. Inconclusive runs are not promotion-eligible.

`tools\triangle_sprt.py` implements Fishtest's logistic Elo likelihood-ratio
calculation, zero-bin regularization, and dynamic overshoot-adjusted stopping
boundaries using only Python's standard library. The LLR and overshoot history
are rebuilt in pair order from the retained PGNs on resume. TriangleTest pins
the analyzer hash and Python version in the run manifest. For the reference
implementation and policy details, see the
[Fishtest SPRT implementation](https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/sprt.py)
and [Fishtest LLR calculations](https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/LLRcalc.py).
Opening selection is randomized per pair using a seed derived from `-Seed`
and the batch's first pair; the same selected opening is played with colors
reversed. Repeated opening selections remain possible, so use a large,
representative opening suite. The analyzer rejects pair PGNs unless games are
complete, engines switch colors, and any supplied FEN tags match.

The combined `paired-match.pgn` is also suitable input for Ordo. For rating a
candidate against a broader opponent pool, include games from the same
conditions in Ordo's PGN input and anchor one opponent with a defensible,
time-control-specific rating. Ordo's result is only comparable to other
results with compatible opponents, openings, hardware, and time controls.

### Calibration and repeatability

Before interpreting strength runs, perform a baseline-versus-identical-baseline
control with `-Stage Calibrate`. It uses the supplied `CurrentParams` file for
both Cutechess players, the same engine binary, and the normal paired SPRT
settings. Use a fresh `RunId` for each independent control; the same ID can be
rerun to resume an interrupted calibration. Calibration runs cannot promote
parameters. Across independent seeded runs, unexpected SPRT passes should be
uncommon at the configured alpha; a single control run cannot establish the
false-positive rate. Repeat candidate comparisons with another seed and
inspect pair PGNs, time forfeits, and score consistency before relying on a
small measured Elo difference.

The manifest and `reproducibility.json` record engine, Cutechess, opening,
parameter, and analyzer hashes; SPRT settings; seed; batch size; clock; Hash;
MoveOverhead; concurrency; and OS, processor, PowerShell, and Python details.
Any mismatch on resume requires a new `RunId`. Bermuda does not yet have a
validated default time control: establish one by checking that several clock
settings complete reliably on the target hardware, with no meaningful
time-forfeit imbalance or concurrency-induced change in results.

### Distributed testing status

TriangleTest remains a local Windows workflow; match batching does not create a
Fishtest server or worker queue. A future distributed runner needs an
authenticated task protocol, worker/build identity, result upload and
integrity checks, and controlled test allocation. For parameter-only
candidates, it must also deliver and verify both weight files on each worker
(Bermuda currently accepts an `EvalParams` path). Until these pieces exist,
combine only games with compatible engine builds, weights, openings, time
controls, and hardware conditions.
