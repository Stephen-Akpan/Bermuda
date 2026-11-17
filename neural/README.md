# Bermuda Graph Residual Evaluation

Graph residuals are optional and default off. The baseline NNUE/MLP path is
unchanged when graph residuals are disabled.

## UCI Options

Load the existing NNUE baseline and the matching graph export, then enable the
graph branch:

```text
setoption name EvalFile value path/to/baseline.nnue
setoption name GraphEvalFile value path/to/graph-model.bmgnn
setoption name GraphScale value 1.0
setoption name GraphEval value true
```

`GraphEval` is a check option and remains false by default. Enabling requires
both a loaded NNUE baseline and GraphEvaluator model. Replacing/clearing the
NNUE or GraphEvaluator file disables graph composition until explicitly enabled
again. `GraphScale` is a finite non-negative runtime multiplier.

M3 graph residuals use the same white-perspective mixed-target probability
units as their baseline. Bermuda's NNUE inference returns centipawns. With graph
mode enabled, the engine computes:

```text
p_combined = clamp(p_baseline_white + GraphScale * graph_residual,
                   1e-6, 1 - 1e-6)
combined_white_cp = eval_scale * logit(p_combined)
combined_stm_cp = combined_white_cp for White to move, otherwise -combined_white_cp
```

The checkpoint's residual multiplier is applied by the C++ GraphEvaluator
before `GraphScale`. A zero `GraphScale` returns the exact existing baseline
score. This probability-to-centipawn conversion is required because the M3
residual is not itself measured in centipawns.

## Parity

With the `nnue-training` virtual environment active, run:

```powershell
.\nnue-training\.venv\Scripts\python.exe -m unittest neural.tests.test_cpp_parity
```

The test builds `graph_eval_parity`, exports a deterministic PyTorch model,
then checks opening, middlegame, endgame, en-passant/castling, and pinned-rook
positions. It reports maximum and mean output error separately from exact
GraphEncoder tensor differences.

## Benchmarks

Build and run the C++ benchmark with the matching NNUE baseline and GraphEvaluator
export:

```powershell
xmake build graph_eval_benchmark
.\build\windows\x64\release\graph_eval_benchmark.exe `
  path\to\baseline.nnue path\to\graph-model.bmgnn 1000 8 2
```

The report separates GraphEncoder time, graph-only inference on pre-encoded
features, combined static evaluation, and fixed-depth full search. Arguments
are iteration count, search depth, and full-search repetitions. Run parity
before interpreting performance or beginning strength tuning.

The benchmark also reports per-stage GraphEvaluator timings in a separate
instrumented pass. Use the uninstrumented graph-inference result for latency;
use the stage breakdown to locate hotspots, since timing instrumentation adds
small overhead.

It compares the reference inference path with an experimental path that
aggregates each target's normalized neighbor hidden vectors before applying the
attack/defence transforms. Both paths use the same encoded features and model
weights; the reference path remains the default. The report includes output
error, per-stage timings, and full-search NPS for each path.

The per-stage profile pass separates individual transform loops for
measurement, so its stage times are diagnostic and are not additive to the
unprofiled end-to-end inference latency.
