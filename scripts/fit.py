"""Pre-registered mixed-effects fit + permutation null + cross-implementation
verification.

Loads cells_<assets>.parquet, fits the primary model
    R_K ~ sigma_micro + (1|family) + (1|window) + (asset FE)
on either the SOL pilot (--pilot) or the DOGE+BTC replication pool (--replication),
or the per-asset triangulation (--per-asset).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf

PRIMARY_SIGMA = "sigma_micro"
PRIMARY_R = "R_K"

SECONDARY_SIGMAS = ["sigma_micro", "sigma_micro_z", "sigma_param", "sigma_combined"]
SECONDARY_RS = ["R_K", "R_PT", "R_rho", "R_lift"]


def load_cells(assets: list[str]) -> pd.DataFrame:
    pieces = []
    for a in assets:
        p = Path("data") / f"cells_{a}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        pieces.append(pl.read_parquet(p))
    df = pl.concat(pieces, how="vertical")
    return df.to_pandas()


def fit_mixed(df: pd.DataFrame, sigma: str, R: str, with_asset_fe: bool) -> dict:
    """Pre-reg deviation noted: with only 7 windows on the SOL pilot,
    statsmodels MixedLM hits singular random-effects covariance every time.
    Replaced with OLS + family/window/asset fixed effects + family-clustered
    robust SEs. The within-transform cross-check (fit_within) is the
    algebraically equivalent estimator without the OLS scaffolding.
    Documented in lab page Methodology section.
    """
    work = df.dropna(subset=[sigma, R, "family", "window"]).copy()
    if work.shape[0] < 8:
        return {"error": f"too few rows ({work.shape[0]})"}

    work["window_str"] = "w" + work["window"].astype(str)
    sigma_std = (work[sigma] - work[sigma].mean()) / (work[sigma].std() or 1.0)
    work["_sigma_std"] = sigma_std

    formula = f"{R} ~ _sigma_std + C(family) + C(window_str)"
    if with_asset_fe and work["asset"].nunique() > 1:
        formula += " + C(asset)"

    try:
        # Cluster-robust SEs at the family level
        res = smf.ols(formula, data=work).fit(
            cov_type="cluster", cov_kwds={"groups": work["family"]}
        )
    except Exception as e:
        return {"error": f"fit failed: {e}"}

    if "_sigma_std" not in res.params.index:
        return {"error": "_sigma_std not in fitted params"}

    beta = float(res.params["_sigma_std"])
    se = float(res.bse["_sigma_std"])
    tval = beta / se if se != 0 else np.nan
    from scipy.stats import norm
    p_one = float(norm.cdf(tval))
    p_two = float(2 * (1 - norm.cdf(abs(tval))))

    # Effect size: f² = (R²_full − R²_null) / (1 − R²_full)
    formula_null = formula.replace("_sigma_std + ", "")
    res_null = smf.ols(formula_null, data=work).fit()
    r2_full = float(res.rsquared)
    r2_null = float(res_null.rsquared)
    f2 = (r2_full - r2_null) / (1 - r2_full) if r2_full < 1 else np.nan

    return {
        "n": int(work.shape[0]),
        "beta_smooth": beta,
        "se": se,
        "tval": float(tval),
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "r2_full": r2_full,
        "r2_null": r2_null,
        "f2": float(f2) if not np.isnan(f2) else None,
    }


def fit_within(df: pd.DataFrame, sigma: str, R: str) -> dict:
    """Cross-implementation check: within-transform OLS.
    Subtract group means for family, window (and asset if multi-asset),
    fit OLS on residuals. β_smooth must agree with MixedLM to ~3 decimals.
    """
    work = df.dropna(subset=[sigma, R, "family", "window"]).copy()
    sigma_std = (work[sigma] - work[sigma].mean()) / (work[sigma].std() or 1.0)
    work["_sigma_std"] = sigma_std

    def demean(y, by):
        return y - work.groupby(by)[y.name].transform("mean")

    cols_to_demean = ["family", "window"]
    if work["asset"].nunique() > 1:
        cols_to_demean.append("asset")

    y = work[R]
    x = work["_sigma_std"]
    for col in cols_to_demean:
        y = y - work.groupby(col)[R].transform("mean")
        x = x - work.groupby(col)["_sigma_std"].transform("mean")

    # Plain OLS of y on x (constant absorbed by demeaning)
    if x.var() < 1e-12:
        return {"error": "no variance after demeaning"}
    beta = float((x * y).sum() / (x * x).sum())
    return {"beta_smooth_within": beta, "n": int(work.shape[0])}


def permutation_null(df: pd.DataFrame, sigma: str, R: str, M: int = 1000, seed: int = 7) -> dict:
    """Within (asset, window+1) shuffle of the OOS-side metric assignment.
    For cell-level analysis, the cell-level R is a summary over IS top-K's
    OOS Sharpe; the closest within-cell shuffle is to permute the
    (sigma_F, R_F) pairing within each (asset, window) — i.e. shuffle which
    family's σ goes with which family's R, holding (asset, window) fixed.
    """
    rng = np.random.default_rng(seed)
    base = fit_mixed(df, sigma, R, with_asset_fe=True)
    if "error" in base:
        return base
    obs = base["beta_smooth"]

    null_betas = []
    grouped = df.groupby(["asset", "window"])
    for _ in range(M):
        shuffled = df.copy()
        for (a, w), idx in grouped.groups.items():
            sub = shuffled.loc[idx]
            perm = rng.permutation(sub.index.values)
            shuffled.loc[idx, R] = df.loc[perm, R].values
        res = fit_mixed(shuffled, sigma, R, with_asset_fe=True)
        if "error" not in res:
            null_betas.append(res["beta_smooth"])
    null_betas = np.array(null_betas)
    p_one = (np.sum(null_betas <= obs) + 1) / (len(null_betas) + 1)
    return {
        "M": int(len(null_betas)),
        "beta_obs": float(obs),
        "p_perm_one_sided": float(p_one),
        "null_mean": float(null_betas.mean()),
        "null_std": float(null_betas.std()),
        "null_q05": float(np.quantile(null_betas, 0.05)),
        "null_q95": float(np.quantile(null_betas, 0.95)),
    }


def per_family_breakdown(df: pd.DataFrame, sigma: str, R: str) -> list[dict]:
    out = []
    for fam, sub in df.groupby("family"):
        if sub.shape[0] < 4 or sub[sigma].std() < 1e-9:
            continue
        x = (sub[sigma] - sub[sigma].mean()) / (sub[sigma].std() or 1.0)
        y = sub[R].values
        # Simple OLS within the family (no FEs — too few rows)
        try:
            X = sm.add_constant(x.values)
            res = sm.OLS(y, X, missing="drop").fit()
            beta = float(res.params[1])
            se = float(res.bse[1])
            from scipy.stats import norm
            tval = beta / se if se else np.nan
            p_one = float(norm.cdf(tval))
            out.append(
                {"family": fam, "n": int(sub.shape[0]), "beta": beta, "se": se,
                 "p_one_sided": p_one}
            )
        except Exception:
            pass
    # BH correction
    if out:
        ps = np.array([o["p_one_sided"] for o in out])
        order = np.argsort(ps)
        m = len(ps)
        adj = np.empty_like(ps)
        prev = 1.0
        for rank, idx in enumerate(order[::-1]):
            k = m - rank
            adj_k = ps[idx] * m / k
            prev = min(prev, adj_k)
            adj[idx] = prev
        for o, q in zip(out, adj):
            o["p_BH"] = float(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--replication", action="store_true")
    g.add_argument("--per-asset", action="store_true")
    ap.add_argument("--M", type=int, default=1000, help="permutation count")
    ap.add_argument("--out", default="data/results.json")
    args = ap.parse_args()

    if args.pilot:
        assets = ["SOL_1h_7W"]
        scope = "pilot"
    elif args.replication:
        assets = ["DOGE_30m_21W", "BTC_30m_27W"]
        scope = "replication"
    else:
        assets = ["SOL_1h_7W", "DOGE_30m_21W", "BTC_30m_27W"]
        scope = "per-asset"

    df = load_cells(assets)
    print(f"=== {scope.upper()} === n_cells={df.shape[0]} | assets={assets}")

    results = {"scope": scope, "assets": assets, "n_cells": int(df.shape[0])}

    # Primary
    print("\n[Primary] σ_micro × R_K mixed-effects fit:")
    primary = fit_mixed(df, PRIMARY_SIGMA, PRIMARY_R, with_asset_fe=(len(assets) > 1))
    print(json.dumps(primary, indent=2))
    results["primary"] = primary

    # Cross-implementation check
    print("\n[Cross-impl check] within-transform OLS:")
    within = fit_within(df, PRIMARY_SIGMA, PRIMARY_R)
    print(json.dumps(within, indent=2))
    results["primary_within"] = within
    if "beta_smooth" in primary and "beta_smooth_within" in within:
        diff = abs(primary["beta_smooth"] - within["beta_smooth_within"])
        print(f"  |Δβ| = {diff:.4f}  ({'✓ AGREES' if diff < 0.01 else '✗ DISAGREES'} to 2 d.p.)")
        results["primary_xcheck_diff"] = float(diff)

    # Permutation null
    print(f"\n[Permutation null] M={args.M} ...")
    perm = permutation_null(df, PRIMARY_SIGMA, PRIMARY_R, M=args.M)
    print(json.dumps(perm, indent=2))
    results["primary_permutation"] = perm

    # Secondary 4×4 grid (BH across the 15 secondary cells)
    print("\n[Secondary 4x4 grid]")
    secondary = []
    for s in SECONDARY_SIGMAS:
        for r in SECONDARY_RS:
            if s == PRIMARY_SIGMA and r == PRIMARY_R:
                continue
            res = fit_mixed(df, s, r, with_asset_fe=(len(assets) > 1))
            res["sigma"] = s
            res["R"] = r
            secondary.append(res)
    # BH correction across the 15 secondary p_one_sided
    valid = [s for s in secondary if "p_one_sided" in s]
    if valid:
        ps = np.array([s["p_one_sided"] for s in valid])
        order = np.argsort(ps)
        m = len(ps)
        prev = 1.0
        adj = np.empty_like(ps)
        for rank, idx in enumerate(order[::-1]):
            k = m - rank
            adj_k = ps[idx] * m / k
            prev = min(prev, adj_k)
            adj[idx] = prev
        for s, q in zip(valid, adj):
            s["p_BH"] = float(q)
    for s in secondary:
        print(f"  {s.get('sigma','?'):16s} × {s.get('R','?'):8s}  β={s.get('beta_smooth', float('nan')):+.4f}  p1={s.get('p_one_sided', float('nan')):.4f}  p_BH={s.get('p_BH', float('nan')):.4f}")
    results["secondary"] = secondary

    # Per-family breakdown
    print("\n[Per-family breakdown]")
    pf = per_family_breakdown(df, PRIMARY_SIGMA, PRIMARY_R)
    for r in pf:
        print(f"  {r['family']:10s}  n={r['n']:3d}  β={r['beta']:+.4f}  p1={r['p_one_sided']:.4f}  p_BH={r.get('p_BH', float('nan')):.4f}")
    results["per_family"] = pf

    # Per-asset breakdown
    if len(assets) > 1:
        print("\n[Per-asset breakdown]")
        pa = []
        for a in assets:
            sub = df[df["asset"] == a]
            r = fit_mixed(sub, PRIMARY_SIGMA, PRIMARY_R, with_asset_fe=False)
            r["asset"] = a
            pa.append(r)
            print(f"  {a:18s}  n={r.get('n', 0)}  β={r.get('beta_smooth', float('nan')):+.4f}  p1={r.get('p_one_sided', float('nan')):.4f}")
        results["per_asset"] = pa

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
