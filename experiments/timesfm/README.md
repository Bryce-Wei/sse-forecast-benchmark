# TimesFM 2.5 / LSTM32 reproduction

Run from the repository root after installing dependencies and fetching the
historical data described in the root README.

```bash
python experiments/timesfm/download_model.py
python experiments/timesfm/run_timesfm.py --smoke
python experiments/timesfm/run_all.py --limit 1
python experiments/timesfm/run_all.py
# Optional English labels
python experiments/timesfm/run_all.py --language en
```

The downloader fetches Google's official `google/timesfm-2.5-200m-pytorch`
checkpoint at revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. It verifies
the config, model card, and 925,181,104-byte weight file against pinned SHA-256
hashes. The default location is ignored `.cache/timesfm/`. To reuse an existing
copy, set `SSE_TIMESFM_CHECKPOINT` to its directory. Run
`python experiments/timesfm/download_model.py --verify-only` to check it
without downloading anything.

`--smoke` performs a real one-date forecast and verifies it after loading a
fresh copy of the model. `--limit 1` runs only the first date; the LSTM still
uses its full 40-epoch selection protocol. `--inputs-only` validates past-only
TimesFM inputs without loading the checkpoint. Full aggregation requires all
64 dates and writes `runs/timesfm/{metrics.json,predictions.json,comparison.png}`.

The paired workflow directly calls the independent
[`experiments/lstm/run_all.py`](../lstm/run_all.py) with `--history 32`.
The single shared training implementation writes to `runs/lstm/history_32/`;
aggregation verifies those records by date, input length, data fingerprint,
and training protocol. Old `runs/timesfm/per_day/lstm/` files are no longer
read. `run_lstm.py` remains only as a compatibility wrapper. The independent
and paired workflows therefore reuse the same completed LSTM days.

The review covers 2026-06-15 through 2026-09-11. Both models receive the same
32 real daily closes and predict one next trading observation. TimesFM uses
the official `timesfm==2.0.1` Python package and frozen weights, with no local
training or fine-tuning. Its official point output is the median (q0.5).
`max_horizon=128` is a model capacity setting; requested forecast horizon is 1.

LSTM uses the preceding two calendar years of closes on each target date,
`LSTM(32) -> LSTM(16, relu) -> Dense(1)`, and a 40-target internal validation
tail to select 1–40 training epochs. It then discards the selection model and
optimizer and refits a fresh model on all past data. Scalers are fitted only
on the appropriate past training partition. Seed 88, MAE, Adam 0.001, batch
32, and clip value 1.0 match the original experiment.

Training information is not matched: TimesFM brings external pretraining,
whereas LSTM trains locally. The fixed checkpoint was public before these
2026 observations; this does not prove that older index data were absent from
pretraining. Review dates were examined in prior experiments and are not an
untouched final test. The left plot is LSTM historical data, not a depiction
of TimesFM's external training corpus.

Plots share the repository layout: actual history on the left, actual and
predicted test values on the right, plus a test-period zoom. Labels support
Chinese and English through `--language`.

`checkpoint_provenance.json` records the reference checkpoint identity, not
an assertion that a new installation was audited. Fresh runs hash weights,
verify frozen state before/after inference, save every input date and value,
and reload first/middle/last forecasts. No weights, old installation logs,
or old trained LSTM models are distributed in this repository.
