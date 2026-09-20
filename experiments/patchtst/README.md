# PatchTST / LSTM20 reproduction

Run commands from the repository root after installing the requirements and
fetching the historical data as described in the root README.

```bash
python experiments/patchtst/run_patchtst.py --smoke
python experiments/patchtst/run_all.py --limit 1
python experiments/patchtst/run_all.py
# Optional English labels
python experiments/patchtst/run_all.py --language en
```

`--smoke` verifies the upstream source hashes, forward/backward calculation,
output shape, and batch independence. `--limit 1` runs the first review date
only, preserving all 40 validation epochs and the selected-epoch refit. It is
not a full benchmark and does not generate overall metrics. The full command
resumes completed dates, checks saved-model reloads, joins the actual closes,
and creates `runs/patchtst/metrics.json`, `predictions.json`, and `comparison.png`.

The paired workflow directly calls the independent
[`experiments/lstm/run_all.py`](../lstm/run_all.py) with `--history 20`.
LSTM training exists only in `experiments/lstm/train.py`; its models and daily
records are stored once in `runs/lstm/history_20/`. Aggregation verifies the
shared results by date, input length, data fingerprint, and training protocol.
Old `runs/patchtst/per_day/lstm/` records are no longer consumed. The old
`run_lstm.py` command is a thin compatibility wrapper around the shared entry.

Protocol: 64 target dates from 2026-06-15 through 2026-09-11; each target uses
only its preceding two calendar years of closes. Both models receive 20
consecutive observations and forecast the next trading observation. The last
40 historical target dates select an epoch from 1–40. A fresh model and
optimizer then refit the full past window for that many epochs. Validation
scalers exclude validation observations. Seed 88, MAE loss, Adam 0.001,
batch 32, and gradient clip value 1.0 reproduce the original experiment.

PatchTST uses the unmodified pinned supervised implementation with 2 layers,
4 attention heads, width 32, patch length 5, stride 3, 6 patches, LayerNorm,
and affine RevIN (17,667 parameters). The classroom-style LSTM uses
`LSTM(32) -> LSTM(16, relu) -> Dense(1)` (7,505 parameters).

These are matched information/training-budget rules, not an isolated causal
test of attention: normalization and model structure differ. This is a
retrospective review of previously examined dates, not a newly untouched test.

The portable scripts adapt the original experiment by replacing machine paths,
adding limited verification, saving relative model paths, and drawing portable
plots. Charts show actual history on the left, actual/predicted test values on
the right, and a test-period zoom; labels support Chinese and English.
The scripts do not retune the model. Historical published results are
separate from new runs; no old success log is used as proof of a new execution.
Different framework versions or hardware can change numerical results.

The vendored model retains its original Apache 2.0 license; see
[`third_party/PatchTST/PROVENANCE.md`](../../third_party/PatchTST/PROVENANCE.md).
