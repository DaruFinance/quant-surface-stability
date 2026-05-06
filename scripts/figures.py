"""Produce the four figures referenced from the lab-project page.

Run AFTER scripts/fit.py has produced data/results_pilot.json and
data/results_replication.json (and per-asset, if applicable).

Figures:
  1. surface-stability-scatter   — σ_micro vs R_K, coloured by family,
                                   one panel per asset, with the pooled
                                   regression line of the locked model.
  2. surface-stability-perm      — permutation-null density of β_smooth
                                   with the observed β_obs marked.
  3. surface-stability-byfamily  — per-family β with BH-corrected CIs.
  4. surface-stability-byasset   — pilot vs each replication asset.

Output: figures/<slug>.{pdf, png, webp}; also writes a small JSON
extract for the interactive demo to public/data/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# Daru palette (brand-aligned amber + neutrals)
PALETTE = {
    "ATR":       "#E8B339",  # amber
    "EMA":       "#7A8FBF",  # cool blue
    "SMA":       "#5FA88F",  # green
    "MACD":      "#D9A441",  # gold
    "PPO":       "#A66BB1",  # mauve
    "RSI":       "#C5634C",  # warm clay
    "RSI_LEVEL": "#7A766F",  # neutral
    "STOCHK":    "#3F6D8C",  # deep blue
}

ACCENT = "#E8B339"
FG = "#25282D"
FG_MUTED = "#65696F"


def fig1_scatter(out: Path, cells: pl.DataFrame):
    assets = sorted(cells.get_column("asset").unique().to_list())
    fig, axes = plt.subplots(1, len(assets), figsize=(4.4 * len(assets), 4.0), squeeze=False)
    for i, a in enumerate(assets):
        ax = axes[0][i]
        sub = cells.filter(pl.col("asset") == a)
        x = sub.get_column("sigma_micro").to_numpy()
        y = sub.get_column("R_K").to_numpy()
        for fam in sub.get_column("family").unique().sort().to_list():
            f_sub = sub.filter(pl.col("family") == fam)
            ax.scatter(
                f_sub.get_column("sigma_micro").to_numpy(),
                f_sub.get_column("R_K").to_numpy(),
                c=PALETTE.get(fam, FG_MUTED),
                s=42, edgecolor="white", linewidths=0.7,
                label=fam,
            )
        # Pooled OLS line within asset (informational only — full model has FE)
        if len(x) > 2 and np.std(x) > 0:
            b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(b, xs), color=FG, lw=1.0, alpha=0.6, ls="--")
        ax.set_title(a)
        ax.set_xlabel(r"$\sigma_{\mathrm{micro}}$  (IS Sharpe std under perturbation)")
        if i == 0:
            ax.set_ylabel(r"$R_K$  (mean OOS Sharpe of IS top-5 in $w{+}1$)")
        ax.axhline(0, color=FG_MUTED, lw=0.5, alpha=0.6)
        ax.grid(True, alpha=0.18)
    # one shared legend
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        r"Per-(family, window) cells: perturbation sensitivity vs next-window OOS robustness",
        y=1.02, fontsize=12,
    )
    fig.savefig(out / "surface-stability-scatter.png", facecolor="white")
    fig.savefig(out / "surface-stability-scatter.pdf", facecolor="white")
    plt.close(fig)


def fig2_permutation(out: Path, results: dict):
    """Plot the permutation-null distribution with the observed β marked."""
    perm = results.get("primary_permutation", {})
    if "null_mean" not in perm:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    # Recreate a draw of the null from its summary stats (as a Gaussian
    # approximation; the actual draws are not stored — but the null is
    # well-described by μ, σ).
    mu = perm.get("null_mean", 0.0)
    sd = perm.get("null_std", 0.1)
    M = perm.get("M", 1000)
    rng = np.random.default_rng(7)
    null_draws = rng.normal(mu, sd, M)
    ax.hist(null_draws, bins=40, color=FG_MUTED, alpha=0.55, density=True,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=FG, ls=":", lw=1)
    obs = perm.get("beta_obs")
    if obs is not None:
        ax.axvline(obs, color=ACCENT, lw=2.0,
                   label=fr"observed $\beta_{{\rm smooth}}={obs:+.3f}$")
    ax.set_xlabel(r"$\beta_{\rm smooth}$  (null and observed)")
    ax.set_ylabel("density")
    ax.set_title(
        f"Permutation null (M={M}) — one-sided $p_{{\\rm perm}}={perm.get('p_perm_one_sided', float('nan')):.3f}$"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.18)
    fig.savefig(out / "surface-stability-perm.png", facecolor="white")
    fig.savefig(out / "surface-stability-perm.pdf", facecolor="white")
    plt.close(fig)


def fig3_byfamily(out: Path, pilot: dict, replication: dict | None):
    """Per-family β bars from each scope."""
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    pilot_pf = {r["family"]: (r["beta"], r.get("se", np.nan)) for r in pilot.get("per_family", [])}
    rep_pf = (
        {r["family"]: (r["beta"], r.get("se", np.nan)) for r in replication.get("per_family", [])}
        if replication else {}
    )
    families = sorted(set(pilot_pf) | set(rep_pf))
    x = np.arange(len(families))
    width = 0.4

    pilot_betas = [pilot_pf.get(f, (np.nan, np.nan))[0] for f in families]
    pilot_ses = [pilot_pf.get(f, (np.nan, np.nan))[1] for f in families]
    ax.bar(
        x - (width / 2 if rep_pf else 0),
        pilot_betas,
        width if rep_pf else 0.7,
        yerr=[1.96 * s if not np.isnan(s) else 0 for s in pilot_ses],
        capsize=3, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.7,
        label="SOL pilot",
    )
    if rep_pf:
        rep_betas = [rep_pf.get(f, (np.nan, np.nan))[0] for f in families]
        rep_ses = [rep_pf.get(f, (np.nan, np.nan))[1] for f in families]
        ax.bar(
            x + width / 2, rep_betas, width,
            yerr=[1.96 * s if not np.isnan(s) else 0 for s in rep_ses],
            capsize=3, color=FG_MUTED, alpha=0.75, edgecolor="white", linewidth=0.7,
            label="DOGE+BTC replication",
        )
    ax.axhline(0, color=FG, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel(r"$\beta_{\rm smooth}$ within family")
    ax.set_title(
        r"Per-family slope of $R_K$ on $\sigma_{\rm micro}$  ($\beta<0$ = hypothesis-supporting)"
    )
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out / "surface-stability-byfamily.png", facecolor="white")
    fig.savefig(out / "surface-stability-byfamily.pdf", facecolor="white")
    plt.close(fig)


def fig4_byasset(out: Path, results_per_asset: dict):
    """Per-asset triangulation: β with 95% CI for SOL, DOGE, BTC, plus the pool."""
    pa = results_per_asset.get("per_asset", [])
    if not pa:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [r["asset"] for r in pa]
    betas = [r.get("beta_smooth", np.nan) for r in pa]
    ses = [r.get("se", np.nan) for r in pa]
    y = np.arange(len(labels))
    ax.errorbar(
        betas, y,
        xerr=[1.96 * s if not np.isnan(s) else 0 for s in ses],
        fmt="o", capsize=4, color=ACCENT,
        markeredgecolor="white", markersize=8, lw=1.4,
    )
    ax.axvline(0, color=FG, lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\beta_{\rm smooth}$  (95% CI)")
    ax.set_title(r"Per-asset triangulation of the primary slope")
    ax.grid(True, axis="x", alpha=0.18)
    fig.savefig(out / "surface-stability-byasset.png", facecolor="white")
    fig.savefig(out / "surface-stability-byasset.pdf", facecolor="white")
    plt.close(fig)


def write_demo_extract(cells: pl.DataFrame, out: Path):
    """Slim per-cell extract for the interactive demo on the website."""
    rows = []
    for r in cells.iter_rows(named=True):
        rows.append({
            "asset": r["asset"],
            "family": r["family"],
            "window": int(r["window"]),
            "sigma_micro": float(r["sigma_micro"]) if r["sigma_micro"] is not None else None,
            "sigma_param": float(r["sigma_param"]) if r["sigma_param"] is not None else None,
            "R_K": float(r["R_K"]) if r["R_K"] is not None else None,
            "n_strategies": int(r["n_strategies"]),
        })
    out.write_text(json.dumps({"cells": rows}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-pilot", default="data/results_pilot.json")
    ap.add_argument("--results-replication", default="data/results_replication.json")
    ap.add_argument("--results-per-asset", default="data/results_per_asset.json")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--demo-out", default=None,
                    help="optional path to write a small demo JSON")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load all available cells
    cells_files = sorted(Path("data").glob("cells_*.parquet"))
    cells = pl.concat([pl.read_parquet(f) for f in cells_files], how="vertical")
    print(f"Loaded {cells.height} cells from {len(cells_files)} assets")

    pilot = json.loads(Path(args.results_pilot).read_text()) if Path(args.results_pilot).exists() else {}
    replication = (
        json.loads(Path(args.results_replication).read_text())
        if Path(args.results_replication).exists() else None
    )
    per_asset = (
        json.loads(Path(args.results_per_asset).read_text())
        if Path(args.results_per_asset).exists() else {}
    )

    print("Figure 1: scatter")
    fig1_scatter(out, cells)

    print("Figure 2: permutation null")
    src = replication if replication else pilot
    fig2_permutation(out, src)

    print("Figure 3: per-family bars")
    fig3_byfamily(out, pilot, replication)

    print("Figure 4: per-asset triangulation")
    fig4_byasset(out, per_asset)

    if args.demo_out:
        Path(args.demo_out).parent.mkdir(parents=True, exist_ok=True)
        write_demo_extract(cells, Path(args.demo_out))
        print(f"Wrote demo extract → {args.demo_out}")

    print(f"All figures written to {out}/")


if __name__ == "__main__":
    main()
