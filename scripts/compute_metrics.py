"""Compute σ (perturbation-sensitivity) and R (OOS robustness) metrics
per (family, asset, window) cell, exactly as pre-registered in
analysis-plan.md.

Inputs:
    data/<asset>.parquet — parser output

Outputs:
    data/cells_<asset>.parquet — (family, asset, window) → σ_*, R_*
    data/strats_<asset>.parquet — (strategy, window) row-level (for sensitivity)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


PERTURBATIONS = ["base", "ENT", "FEE", "SLI", "ENT+IND"]
K_PRIMARY = 5
MIN_STRATS_PER_CELL = 30
MIN_NEIGHBORHOOD = 5


def load(asset: str) -> pl.DataFrame:
    p = Path("data") / f"{asset}.parquet"
    df = pl.read_parquet(p)
    n_in = df.height
    # Drop NaN/inf
    df = df.filter(pl.col("sharpe").is_finite() & pl.col("pf").is_finite())
    # Discovered post-pre-reg: the source backtester emits Sharpe values up
    # to 1.44e16 for 2-trade outcomes where the within-window return std is
    # essentially zero, producing a divide-by-near-zero singularity. These
    # are not real outcomes; they are numerical artifacts that would
    # contaminate every aggregate (mean, std, top-K). Filter |Sharpe| > 100,
    # which is far beyond the worst non-artifact value observed in the
    # corpus (~|66|). This is a deviation from the locked pre-reg's sentinel
    # rule and is reported in the lab page as exploratory.
    df = df.filter(pl.col("sharpe").abs() <= 100.0)
    n_out = df.height
    print(f"  filter (sentinel + |Sharpe| ≤ 100): {n_in} -> {n_out} ({n_in - n_out} dropped)")
    return df


def strategy_pivot(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot to one row per (strategy, window): IS_base, IS_ENT, IS_FEE, IS_SLI,
    IS_ENT+IND, OOS_base, OOS_PF (PF helps R_PT), trades_base.
    """
    df = df.with_columns(
        (pl.col("sample") + "_" + pl.col("perturbation")).alias("sp"),
    )
    # we need: IS_base, IS_ENT, IS_FEE, IS_SLI, IS_ENT+IND, OOS_base, OOS_base_pf
    sharpe_pivot = df.pivot(
        on="sp", index=["asset", "family", "strategy", "base_param", "transformation",
                          "confluence", "sl_regime", "window"],
        values="sharpe", aggregate_function="first",
    )
    # Add OOS PF separately for R_PT
    pf_pivot = df.pivot(
        on="sp", index=["strategy", "window"],
        values="pf", aggregate_function="first",
    ).select(
        "strategy", "window",
        pl.col("OOS_base").alias("OOS_pf"),
    )
    out = sharpe_pivot.join(pf_pivot, on=["strategy", "window"], how="left")

    # σ_micro = std across {IS_base, IS_ENT, IS_FEE, IS_SLI, IS_ENT+IND}
    is_cols = ["IS_base", "IS_ENT", "IS_FEE", "IS_SLI", "IS_ENT+IND"]
    available = [c for c in is_cols if c in out.columns]
    if len(available) < 2:
        raise RuntimeError(f"need ≥2 perturbation columns; got {available}")
    out = out.with_columns(
        pl.concat_list(available).alias("_is_perts"),
    )
    out = out.with_columns(
        pl.col("_is_perts").list.std().alias("sigma_micro"),
        pl.col("_is_perts").list.len().alias("_n_perts"),
    ).drop("_is_perts")
    return out


def neighborhood_param_std(piv: pl.DataFrame) -> pl.DataFrame:
    """σ_param: for each (strategy, window), std of IS_base across the 1-step
    parameter neighbourhood — strategies sharing (family, base_param, sl_regime)
    with this strategy but differing in transformation OR confluence.

    Strict 1-step is impossible to define cleanly without an ordering on
    transformations / confluences, so we follow the pre-reg's looser
    definition: same (family, base_param, sl_regime), any (transformation,
    confluence). This is documented in analysis-plan-locked.md as the
    post-pilot lock rule.
    """
    # neighborhood key = (family, base_param, sl_regime, window)
    nbhd = (
        piv.group_by(["family", "base_param", "sl_regime", "window"])
        .agg(
            pl.col("IS_base").std().alias("sigma_param"),
            pl.col("IS_base").len().alias("nbhd_n"),
        )
    )
    return piv.join(
        nbhd, on=["family", "base_param", "sl_regime", "window"], how="left"
    )


def cell_metrics(piv: pl.DataFrame, asset: str) -> pl.DataFrame:
    """Aggregate to (family, asset, window) cells and compute σ_F, R_K, R_PT,
    R_ρ, R_lift for transitions (window → window+1)."""

    # Build per-window rank tables to compute IS rank and OOS rank
    rows = []
    families = sorted(piv.get_column("family").unique().to_list())
    n_windows = int(piv.get_column("window").max())  # type: ignore

    for fam in families:
        sub = piv.filter(pl.col("family") == fam)
        for w in range(1, n_windows):
            cur = sub.filter(pl.col("window") == w).filter(
                pl.col("IS_base").is_finite()
            )
            nxt = sub.filter(pl.col("window") == w + 1).filter(
                pl.col("OOS_base").is_finite()
            )
            if cur.height < MIN_STRATS_PER_CELL or nxt.height < MIN_STRATS_PER_CELL:
                continue
            # Join cur (IS) to nxt (OOS) on strategy
            joined = cur.select(
                "strategy", "IS_base", "sigma_micro", "sigma_param", "_n_perts"
            ).join(
                nxt.select("strategy", "OOS_base", "OOS_pf").rename(
                    {"OOS_base": "OOS_next", "OOS_pf": "OOS_pf_next"}
                ),
                on="strategy",
                how="inner",
            )
            joined = joined.filter(
                pl.col("OOS_next").is_finite() & pl.col("IS_base").is_finite()
            )
            if joined.height < MIN_STRATS_PER_CELL:
                continue

            # IS rank-by-Sharpe, descending
            ranked = joined.sort("IS_base", descending=True)
            top_k = ranked.head(K_PRIMARY)
            top_decile_n = max(1, ranked.height // 10)
            top_decile = ranked.head(top_decile_n)

            # σ at the cell level: mean of σ_micro / σ_param over IS top-K
            sigma_micro_cell = top_k.get_column("sigma_micro").mean()
            sigma_param_cell = top_k.get_column("sigma_param").mean()
            # σ_combined = sqrt(σ_micro² + σ_param²) at top-K mean level
            sigma_combined_cell = float(
                np.sqrt((sigma_micro_cell or 0.0) ** 2 + (sigma_param_cell or 0.0) ** 2)
            )
            # σ_micro_z = same as σ_micro but z-normalised within (family, asset, window)
            mu = ranked.get_column("sigma_micro").mean() or 0.0
            sd = ranked.get_column("sigma_micro").std() or 1.0
            if sd == 0 or sd is None:
                sd = 1.0
            sigma_micro_z_cell = (sigma_micro_cell - mu) / sd if sigma_micro_cell is not None else None

            # R_K: mean OOS Sharpe of IS top-K
            R_K = top_k.get_column("OOS_next").mean()
            # R_PT: fraction of IS top-K with OOS PF > 1
            R_PT = top_k.get_column("OOS_pf_next").gt(1.0).mean()
            # R_ρ: Spearman rank correlation IS rank vs OOS rank over the cell
            is_rank = ranked.get_column("IS_base").rank().to_numpy()
            oos_rank = ranked.get_column("OOS_next").rank().to_numpy()
            R_rho = float(np.corrcoef(is_rank, oos_rank)[0, 1]) if len(is_rank) > 2 else None
            # R_lift: mean OOS Sharpe of IS top-decile / mean OOS Sharpe of full pool
            R_lift_num = top_decile.get_column("OOS_next").mean()
            R_lift_den = ranked.get_column("OOS_next").mean()
            R_lift = (
                float(R_lift_num / R_lift_den)
                if R_lift_den is not None and abs(R_lift_den) > 1e-9
                else None
            )

            rows.append({
                "asset": asset,
                "family": fam,
                "window": w,  # transition: w → w+1
                "n_strategies": joined.height,
                "sigma_micro": sigma_micro_cell,
                "sigma_micro_z": sigma_micro_z_cell,
                "sigma_param": sigma_param_cell,
                "sigma_combined": sigma_combined_cell,
                "R_K": R_K,
                "R_PT": R_PT,
                "R_rho": R_rho,
                "R_lift": R_lift,
                "is_top_k_mean": top_k.get_column("IS_base").mean(),
            })

    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    args = ap.parse_args()

    print(f"Loading data/{args.asset}.parquet ...")
    df = load(args.asset)
    print(f"  rows: {df.height}")

    print("Pivoting to per-(strategy, window) ...")
    piv = strategy_pivot(df)
    print(f"  rows: {piv.height}")

    print("Adding σ_param neighbourhood ...")
    piv = neighborhood_param_std(piv)

    print("Aggregating to cells ...")
    cells = cell_metrics(piv, args.asset)
    print(f"  cells: {cells.height}")
    print(cells)

    out_cells = Path("data") / f"cells_{args.asset}.parquet"
    out_strats = Path("data") / f"strats_{args.asset}.parquet"
    cells.write_parquet(out_cells)
    piv.write_parquet(out_strats)
    print(f"Wrote {out_cells}, {out_strats}")


if __name__ == "__main__":
    main()
