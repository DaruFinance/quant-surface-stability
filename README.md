# quant-surface-stability

> The pre-registered study behind [Does in-sample smoothness predict out-of-sample skill?](https://daru.finance/projects/strategy-surface-stability), by Daniel Gatto. Full write-up on [daru.finance](https://daru.finance).

Pre-registered empirical study of the claim:

> *"Smooth, broad ridges mean the family has a stable performance basin under perturbation of its parameters. Sharp spikes flanked by collapse mean the family is brittle — its in-sample peaks are likely overfit rather than real."*

The pre-registration in `analysis-plan.md` is hash-locked at first commit; any deviation in the executed analysis is reported as exploratory.

## Layout

```
analysis-plan.md           # locked pre-registration
analysis-plan-locked.md    # post-pilot hyperparameter lock (created after SOL pilot)
src/parse_corpus.rs        # Rust parser: per-strategy .txt → parquet
scripts/compute_metrics.py # σ and R metrics on parsed parquet
scripts/fit.py             # mixed-effects + permutation null
data/                      # parquet outputs (gitignored)
figures/                   # final figures
```

## Reproducibility

```bash
# Parse SOL (pilot)
cargo run --release --bin parse_corpus -- --asset SOL_1h_7W

# Pilot analysis
python3 scripts/compute_metrics.py --asset SOL_1h_7W
python3 scripts/fit.py --pilot

# After hyperparameter lock:
cargo run --release --bin parse_corpus -- --asset DOGE_30m_21W
cargo run --release --bin parse_corpus -- --asset BTC_30m_27W
python3 scripts/compute_metrics.py --asset DOGE_30m_21W
python3 scripts/compute_metrics.py --asset BTC_30m_27W
python3 scripts/fit.py --replication
```
