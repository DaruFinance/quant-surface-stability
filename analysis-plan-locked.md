# Locked hyperparameters — post-pilot, pre-replication

**Locked at**: end of SOL_1h_7W pilot run, before any DOGE_30m_21W or BTC_30m_27W parquet exists.
**Pre-reg this locks against**: `analysis-plan.md` first commit at `c451177f21efa5b00263ff422dccd46eb313d8b4`.

## Hyperparameters (final)

| symbol / rule | value | locked from pre-reg / decided in pilot |
|---|---|---|
| K (top-K for R_K, R_PT) | 5 | pre-reg |
| M (permutation count) | 1000 | pre-reg |
| α primary | 0.05 (one-sided) | pre-reg |
| α replication | 0.01 (one-sided) | pre-reg |
| f² floor for article elevation | 0.04 | pre-reg |
| min strategies / cell | 30 | pre-reg |
| min N₁ for σ_param | 5 | pre-reg |
| σ_param neighbourhood definition | strategies sharing (family, base_param, sl_regime) — vary transformation **or** confluence | pilot lock |
| |Sharpe| numerical artifact cap | drop rows with `|Sharpe| > 100` | pilot lock — documented exploratory deviation from pre-reg sentinel rule, justified by 1.4e16 backtester divide-by-near-zero singularities at trades=2 |
| Estimator | OLS with cluster-robust SEs (clusters = family), fixed effects on family + window + asset | pilot lock — documented exploratory deviation from MixedLM pre-reg, forced by singular random-effects covariance at n_windows=7 |
| Cross-implementation check | within-transform OLS (analytical demeaning) | pilot lock (replaces the absent R `lme4` cross-check) |

## Pilot summary (informational; replication ignores this)

- n_cells = 42 (7 families × 6 transitions; RSI_LEVEL excluded by min-strategies floor)
- β_smooth = +0.115 (standardised σ_micro)
- se = 0.121, t = +0.95
- p_one_sided (H₁: β<0) = 0.829
- p_perm (M=1000) = 0.843
- f² = 0.038
- Cross-implementation: β_within = +0.115 (Δ = 1e-13)
- Null-permutation calibration: null_mean = -0.004, null_std = 0.118, null_q05 = -0.20, null_q95 = +0.19 — clean
- Per-family: ATR β=-0.33 p_BH<0.0001 (strong negative, hypothesis-supporting); RSI β=-0.38 p_BH=0.28; PPO/SMA weakly negative; EMA/STOCHK/MACD positive
- Verdict: H₀ retained on SOL pilot. Pilot was for hyperparameter locking, not confirmation; replication on DOGE+BTC proceeds.
