# Pre-registered analysis plan — surface stability as a robustness signal

**Status**: Locked at first commit. Any deviation from this document is reported in the Results section as "exploratory" and the deviation is justified explicitly.

**Context**: The Daru Finance `/corpus` page contains the assertion that *"smooth, broad ridges mean the family has a stable performance basin under perturbation of its parameters; sharp spikes flanked by collapse mean the family is brittle — its in-sample peaks are likely overfit rather than real."* This document pre-registers the empirical test of that assertion.

**Note on revision from initial design**: The initial plan committed to a 2D Sharpe-surface methodology. Pre-data inspection of the corpus directory layout revealed (i) most families have a 1D numeric parameter axis only, not 2D; and (ii) every strategy ships with a precomputed microstructural perturbation suite per window. The methodology was therefore reformulated **before any data was loaded into the analysis pipeline**: the "perturbation of parameters" in the verbal claim is now operationalised against (a) the empirical perturbation suite already in the data, and (b) 1-step parameter neighbourhoods in the corpus structure. The hypothesis direction and decision rules below are unchanged.

---

## 1. Hypothesis

**Subject claim** (verbal): smoother in-sample Sharpe behaviour under parameter perturbation predicts more stable out-of-sample performance for in-sample-selected strategies.

**Operational form**:
- For strategy θ in the IS top-K of (family F, asset a, window w), let σ(θ) be a perturbation-sensitivity metric (smaller = more stable basin = "smoother").
- Let R(θ) be the OOS Sharpe in window w+1.
- For each (F, a, w) cell, aggregate over IS top-K to get (σ_F, R_F).

**H₀**: After controlling for family/asset/window fixed effects, σ has zero or positive association with R (i.e., perturbation-sensitive strategies do **not** perform worse OOS).

**H₁ (one-sided)**: σ negatively predicts R — IS-stable strategies generalise better OOS.

---

## 2. Variables

### 2.1 Smoothness / sensitivity metrics (4 pre-specified, primary = σ_micro)

| key       | definition |
|-----------|-----------|
| **σ_micro** (primary) | `std({IS_Sharpe(θ, w, p) : p ∈ {base, +ENT, +FEE, +SLI, +ENT+IND}})` — IS Sharpe sensitivity to the precomputed microstructural perturbation suite. |
| σ_micro_z | same as σ_micro but z-normalised within (F, a, w) so cross-family magnitudes are comparable |
| σ_param   | `std({IS_Sharpe(θ', w) : θ' ∈ N₁(θ)})` over the 1-step parameter neighbourhood `N₁(θ)`: strategies sharing (family, base_param, SL_regime) with θ but differing in exactly one of {transformation, confluence}. |
| σ_combined | `√(σ_micro² + σ_param²)` — total perturbation sensitivity |

All four are sensitivity measures (larger = more sensitive = "spikier"). σ_micro is primary; the others are robustness checks.

### 2.2 OOS robustness metrics (4 pre-specified, primary = R_K)

| key  | definition |
|------|-----------|
| **R_K** (primary) | mean base OOS Sharpe in window w+1 of the K=5 strategies with highest base IS Sharpe in window w, restricted to (F, a) |
| R_PT | fraction of IS top-K with OOS profit factor > 1 in window w+1 |
| R_ρ  | Spearman rank correlation between IS Sharpe (w) and OOS Sharpe (w+1) over the full (F, a) population |
| R_lift | mean OOS Sharpe of IS top-decile / mean OOS Sharpe of full (F, a) population, both in window w+1 |

K = 5 matches the existing CheckerWFO portfolio convention. R_K is primary; R_PT/R_ρ/R_lift are robustness checks.

### 2.3 Aggregation level

The primary analysis is at the **(family, asset, window) cell level**:
- σ_F(F, a, w) = mean σ_micro over IS top-K
- R_F(F, a, w) = R_K = mean OOS Sharpe of IS top-K in window w+1

Secondary analysis at the strategy level (n much larger; uses every IS top-K strategy across all cells as a single regression unit) is run as a sensitivity check.

---

## 3. Inclusion / exclusion rules

- **Cells**: (family F, asset a, window w) is included iff (i) window w+1 exists, (ii) ≥ 30 strategies in (F, a) have all five perturbation variants present in window w (data-quality floor), (iii) at least K=5 strategies have non-null IS Sharpe.
- **Strategies for σ_param**: a strategy θ contributes only if its parameter neighbourhood N₁(θ) has ≥ 5 members. Otherwise σ_param is recorded as missing for that θ; aggregation drops missings.
- **Sentinel handling**: rows where IS or OOS Sharpe parses as `nan` / `±inf` are excluded from all metrics. Trade counts < 5 are NOT excluded (they are real outcomes); but the count is recorded as a control variable.
- **Family RSI_LEVEL** has only 12 strategies on SOL. It will be reported but excluded from the primary cell-level model (n < 30 floor); included in the strategy-level sensitivity.

---

## 4. Statistical plan

### 4.1 Primary model (one per (σ, R) pair, 4 × 4 = 16 total; primary = σ_micro × R_K)

```
R_F(F, a, w) = β₀ + β_smooth · σ_F(F, a, w) + γ_F + γ_a + γ_w + ε
```

Linear mixed-effects model with random intercepts for F, a, w (`statsmodels.regression.mixed_linear_model.MixedLM`, REML). Cluster-robust SEs via wild bootstrap with B = 2000, clusters at the family level.

### 4.2 Inference

- **Permutation null** (M = 1000): within (a, w+1), shuffle the OOS Sharpe → strategy assignment, preserving the marginal OOS distribution. Recompute R_K, refit the model, record β_smooth_null. Empirical one-sided p-value = `(#{β_smooth_null ≤ β_smooth_obs} + 1) / (M + 1)`.
- **Multiple-comparisons control**: BH across the 15 secondary (σ, R) pairs. Primary is its own family-of-1.
- **One-sided test** for the primary (claim is directional). Two-sided p-values reported alongside.
- **Effect size**: f² = R²_full − R²_null, where R²_null is the model with γ_F + γ_a + γ_w only.

### 4.3 Pilot–replication protocol

**Pilot (SOL_1h_7W)**:
1. Run all 16 (σ, R) combinations.
2. Decide hyperparameters not pre-specified above by inspection: ε in σ_topfrac if used, K_alt for R_K sensitivity, axis-projection rules in σ_param.
3. **Lock** all hyperparameters in `analysis-plan-locked.md` and commit BEFORE touching DOGE or BTC parquet.

**Replication (DOGE_30m_21W + BTC_30m_27W pooled)**:
1. Apply locked hyperparameters.
2. Fit primary model on pooled DOGE + BTC.
3. Per-asset triangulation (SOL, DOGE, BTC each separately).
4. Per-family BH-corrected breakdown across 8 families.

### 4.4 Pre-registered article-elevation criterion

Replication primary one-sided p_perm < 0.01 **AND** sign-consistent β_smooth (negative in the perturbation→OOS direction) **AND** effect size f² ≥ 0.04.

If all three: lab page + article + corpus-page link strengthens claim.
If replication p < 0.05 and sign-consistent: lab page only; corpus-page reads "weak supporting evidence".
If NS or sign-flipped: lab page reports the null; corpus-page sentence is rewritten.

In all three branches the lab page publishes; only the article is conditional.

### 4.5 Power

Mixed-effects, ICC ≈ 0.3, n_cells: SOL pilot ≈ 7 × 6 = 42; replication pooled ≈ 7 × (20 + 26) = 322. At f² = 0.05, replication has > 0.95 power; pilot ≈ 0.55 (pilot is for hyperparameter locking, not confirmation).

### 4.6 Cross-language verification

R is not available in the current environment. Verification is performed instead by **two independent Python implementations** of the primary model:
1. `statsmodels.MixedLM` (REML, default).
2. Manual within-transform: subtract group means for F, a, w, fit OLS on residuals, compute β_smooth and SEs analytically.
The two estimates of β_smooth must agree to 3 decimal places. If they don't, the analysis is halted and the discrepancy is investigated.

---

## 5. Pipeline

### 5.1 Data parsing (Rust)

`src/parse_corpus.rs`:
- Walk `/mnt/d/Strategies/<asset>/<family>/<strat>/<strat>.txt`.
- For each strategy, parse strategy name → (family, base_param, transformation, confluence, sl_regime).
- For each window `Wxx` line of form `^\s*W(\d+)\s+(IS|OOS)(?:\+(\w+))?\s+\(LB\s+\d+\)\s+\|\s+Trades:\s*(\d+)\s+ROI:\$([\-\d\.]+)\s+PF:\s*([\d\.]+)\s+Shp:\s*([\-\d\.]+)\s+Win:\s*([\d\.]+)%\s+Exp:\$([\-\d\.]+)\s+MaxDD:\$([\-\d\.]+)`, extract: window, sample (IS|OOS), perturbation (None | ENT | FEE | SLI | ENT+IND), trades, ROI, PF, Sharpe, win%, expectancy, MaxDD.
- Output: `data/<asset>.parquet`, schema:
  ```
  asset:        Utf8
  family:       Utf8
  strategy:     Utf8
  base_param:   Utf8
  transformation: Utf8
  confluence:   Utf8
  sl_regime:    Utf8
  window:       Int8
  sample:       Utf8         # "IS" or "OOS"
  perturbation: Utf8         # "base" | "ENT" | "FEE" | "SLI" | "ENT+IND"
  trades:       Int32
  roi:          Float32
  pf:           Float32
  sharpe:       Float32
  win_rate:     Float32
  expectancy:   Float32
  max_dd:       Float32
  ```

### 5.2 Metric computation (Python)

`scripts/compute_metrics.py`:
- Build cell-level dataframe: one row per (family, asset, window), columns σ_micro_F, σ_micro_z_F, σ_param_F, σ_combined_F, R_K, R_PT, R_ρ, R_lift.
- Build strategy-level dataframe: one row per (strategy, window) for the secondary sensitivity analysis.

### 5.3 Inference (Python)

`scripts/fit.py`:
- Primary: σ_micro × R_K mixed-effects fit + permutation null.
- Secondary 16 cells of (σ, R) grid with BH correction.
- Per-asset triangulation.
- Per-family BH breakdown.
- Output: `data/results.parquet`, `figures/*.pdf`.

---

## 6. Hyperparameters fixed in this document (no further tuning allowed for primary)

| symbol | value | rationale |
|--------|-------|-----------|
| K      | 5     | matches existing portfolio convention (CheckerWFO) |
| K_robustness | {3, 5, 10} | sensitivity check only |
| M (perm) | 1000 | standard for permutation null with n ≈ 350 |
| B (boot) | 2000 | wild bootstrap for cluster SEs |
| α primary | 0.05 (one-sided) | pilot decision threshold |
| α replication | 0.01 (one-sided) | article elevation threshold |
| f² floor | 0.04 | small-medium effect for article elevation |
| min strategies / cell | 30 | data-quality floor |
| min N₁ for σ_param | 5 | neighbourhood size floor |

Hyperparameters left for post-pilot lock (in `analysis-plan-locked.md`):
- σ_param neighbourhood definition exact rules (how strict "1-step" is — full transformation set, full confluence set, or restricted to nearest by some lexicographic ordering).
- Bootstrap cluster choice if family clustering proves inadequate.

---

## 7. Decision rules

| outcome | classification | action |
|---------|---------------|--------|
| SOL p_perm < 0.05 AND replication p_perm < 0.01 AND f² ≥ 0.04 AND sign(β_smooth) < 0 | **H₁ supported, strongly** | lab page + article + corpus claim strengthened |
| Replication p_perm < 0.05 AND sign(β_smooth) < 0 | H₁ supported, weakly | lab page only; corpus claim hedged |
| Replication NS or sign(β_smooth) ≥ 0 | H₀ retained | lab page reports the null; corpus sentence rewritten to remove unproven claim |

---

## 8. Verification checklist

1. Parser correctness: 5 random `.txt` files per asset spot-checked by hand against parquet rows.
2. Cross-implementation agreement: `MixedLM` β_smooth and within-transform β_smooth agree to 3 decimal places.
3. Permutation null calibration: under H₀-by-construction (shuffle σ → R link), empirical Type-I rate ≈ nominal α at α ∈ {0.01, 0.05, 0.10}.
4. Replication isolation: directory access logs / parquet mtime show DOGE/BTC was not parsed until after `analysis-plan-locked.md` commit timestamp.
5. Pre-reg integrity: `git log --follow analysis-plan.md` shows the file's first-commit hash exactly matches the hash referenced from the lab page.
