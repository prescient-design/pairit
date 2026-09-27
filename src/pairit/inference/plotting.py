"""Visualization utilities for inference-time compute results.

All functions accept parquet DataFrames directly (loaded from the output
of best-of-N or beam search runs) and return matplotlib Figure objects.
Figures are also saved to disk when ``save_path`` is provided.

Available plots
---------------
1. ``plot_property_vs_round`` — line plot of a property (mean/median/max)
   across beam search rounds, with min-max shading.
2. ``plot_vin_rates_vs_round`` — validity, V+I, and V+I+N rates per round.
3. ``plot_beam_diversity`` — box plot of pairwise Tanimoto distances among
   beam members per round (detects beam collapse).
4. ``plot_trajectories`` — spaghetti plot of property values along lineages,
   one line per trajectory, colored by final score.
5. ``plot_tanimoto_drift`` — how far molecules drift from initial seeds
   over rounds (individual trajectories + mean).
6. ``plot_best_of_n_distribution`` — per-seed histogram of all N candidates
   with the selected molecule highlighted as a vertical line.
7. ``plot_property_pareto`` — two-property scatter with Pareto front
   highlighted (e.g., logP vs SA).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_and_return(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 1. Property vs Round
# ---------------------------------------------------------------------------

def plot_property_vs_round(
    results_df: pd.DataFrame,
    property_name: str,
    agg: str = "selected",
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Line plot of a property across generation rounds.

    Parameters
    ----------
    results_df : pd.DataFrame
        ``all_generations.parquet`` or concatenated round DataFrames.
        Must have columns ``round`` and ``{property_name}``.
    property_name : str
        Property to plot.
    agg : str
        Which molecules to include:
        - ``"selected"``: only selected molecules (beam members).
        - ``"all"``: all generated molecules.
        - ``"valid"``: all valid molecules.
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    df = results_df.copy()
    if agg == "selected":
        df = df[df["selected"] == True]  # noqa: E712
    elif agg == "valid":
        df = df[df["is_valid"] == True]  # noqa: E712

    df = df.dropna(subset=[property_name, "round"])

    grouped = df.groupby("round")[property_name]
    means = grouped.mean()
    medians = grouped.median()
    maxs = grouped.max()
    mins = grouped.min()

    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = means.index

    ax.fill_between(rounds, mins, maxs, alpha=0.15, color="steelblue", label="min-max")
    ax.plot(rounds, means, "o-", color="steelblue", label="mean", linewidth=2)
    ax.plot(rounds, medians, "s--", color="darkorange", label="median", linewidth=1.5)

    ax.set_xlabel("Round")
    ax.set_ylabel(property_name)
    ax.set_title(title or f"{property_name} vs Round ({agg})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. V,I,N Rates vs Round
# ---------------------------------------------------------------------------

def plot_vin_rates_vs_round(
    results_df: pd.DataFrame,
    property_name: str,
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Stacked line plot of validity, V+I, and V+I+N rates per round.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns ``round``, ``is_valid``,
        ``is_improved_{property_name}``, ``is_vin_{property_name}``.
    property_name : str
        Property for improvement/V,I,N check.
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    imp_col = f"is_improved_{property_name}"
    vin_col = f"is_vin_{property_name}"

    df = results_df.copy()
    rounds = sorted(df["round"].dropna().unique())

    validity_rates = []
    vi_rates = []
    vin_rates = []

    for r in rounds:
        rdf = df[df["round"] == r]
        n = len(rdf)
        if n == 0:
            validity_rates.append(0)
            vi_rates.append(0)
            vin_rates.append(0)
            continue

        validity_rates.append(rdf["is_valid"].sum() / n)
        if imp_col in rdf.columns:
            vi = (rdf["is_valid"] & rdf[imp_col]).sum() / n
            vi_rates.append(vi)
        else:
            vi_rates.append(0)
        if vin_col in rdf.columns:
            vin_rates.append(rdf[vin_col].sum() / n)
        else:
            vin_rates.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, validity_rates, "o-", label="Valid", color="tab:blue", linewidth=2)
    ax.plot(rounds, vi_rates, "s-", label="Valid + Improved", color="tab:orange", linewidth=2)
    ax.plot(rounds, vin_rates, "^-", label="V,I,N", color="tab:green", linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Rate")
    ax.set_title(title or f"V,I,N Rates vs Round ({property_name})")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Beam Diversity
# ---------------------------------------------------------------------------

def plot_beam_diversity(
    beam_state_dfs: list[pd.DataFrame] | pd.DataFrame,
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Box plot of pairwise Tanimoto distances among beam members per round.

    Parameters
    ----------
    beam_state_dfs : list[pd.DataFrame] or pd.DataFrame
        Either a list of beam state DataFrames (one per round, from
        ``per_round/beam_state_{i}.parquet``) or a single DataFrame with
        a ``round`` column.
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    if isinstance(beam_state_dfs, pd.DataFrame):
        rounds = sorted(beam_state_dfs["round"].unique())
        dfs = [beam_state_dfs[beam_state_dfs["round"] == r] for r in rounds]
    else:
        dfs = beam_state_dfs
        rounds = list(range(len(dfs)))

    all_distances: list[list[float]] = []

    for rdf in dfs:
        smiles_col = "smiles_canonical" if "smiles_canonical" in rdf.columns else "smiles_generated"
        smiles_list = rdf[smiles_col].dropna().unique().tolist()

        fps = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

        dists = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(1.0 - sim)

        all_distances.append(dists)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(all_distances)))
    bp = ax.boxplot(
        all_distances,
        positions=positions,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")

    ax.set_xlabel("Round")
    ax.set_ylabel("Pairwise Tanimoto Distance")
    ax.set_title(title or "Beam Diversity (Pairwise Tanimoto Distance)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(r) for r in rounds])
    ax.grid(True, alpha=0.3, axis="y")

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Trajectory Spaghetti Plot
# ---------------------------------------------------------------------------

def plot_trajectories(
    trajectories_df: pd.DataFrame,
    property_name: str,
    max_trajectories: int = 50,
    color_by: str = "final_score",
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Spaghetti plot of property values across rounds for each trajectory.

    Parameters
    ----------
    trajectories_df : pd.DataFrame
        From ``trajectories.parquet``. Must have columns matching
        ``round_{N}_{property_name}``.
    property_name : str
        Property to plot.
    max_trajectories : int
        Maximum trajectories to plot (samples randomly if exceeded).
    color_by : str
        ``"final_score"``: color by final property value.
        ``"initial_seed"``: color by initial seed index.
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    df = trajectories_df.copy()
    if len(df) > max_trajectories:
        df = df.sample(n=max_trajectories, random_state=42)

    # Find round columns for this property
    round_cols = sorted(
        [c for c in df.columns if c.startswith("round_") and c.endswith(f"_{property_name}")],
        key=lambda c: int(c.split("_")[1]),
    )

    if not round_cols:
        raise ValueError(
            f"No round columns found for property '{property_name}'. "
            f"Available columns: {[c for c in df.columns if c.startswith('round_')]}"
        )

    rounds = [int(c.split("_")[1]) for c in round_cols]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping
    final_col = round_cols[-1]
    if color_by == "final_score" and final_col in df.columns:
        values = df[final_col].fillna(0)
        norm = plt.Normalize(values.min(), values.max())
        cmap = plt.cm.viridis
    else:
        norm = plt.Normalize(0, len(df))
        cmap = plt.cm.tab20

    for idx, (_, row) in enumerate(df.iterrows()):
        y_values = [row.get(c) for c in round_cols]
        valid_mask = [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y_values]
        x = [r for r, m in zip(rounds, valid_mask) if m]
        y = [v for v, m in zip(y_values, valid_mask) if m]

        if not x:
            continue

        if color_by == "final_score":
            color = cmap(norm(row.get(final_col, 0) or 0))
        else:
            color = cmap(norm(idx))

        ax.plot(x, y, "-o", color=color, alpha=0.5, markersize=3, linewidth=1)

    ax.set_xlabel("Round")
    ax.set_ylabel(property_name)
    ax.set_title(title or f"Trajectories: {property_name} across rounds")
    ax.grid(True, alpha=0.3)

    if color_by == "final_score":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=f"Final {property_name}")

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Tanimoto Drift
# ---------------------------------------------------------------------------

def plot_tanimoto_drift(
    trajectories_df: pd.DataFrame,
    max_trajectories: int = 50,
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot Tanimoto distance from initial seed over rounds.

    Parameters
    ----------
    trajectories_df : pd.DataFrame
        From ``trajectories.parquet``. Must have columns matching
        ``round_{N}_tanimoto_to_seed``.
    max_trajectories : int
        Maximum trajectories to plot.
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    df = trajectories_df.copy()
    if len(df) > max_trajectories:
        df = df.sample(n=max_trajectories, random_state=42)

    tan_cols = sorted(
        [c for c in df.columns if c.startswith("round_") and c.endswith("_tanimoto_to_seed")],
        key=lambda c: int(c.split("_")[1]),
    )

    if not tan_cols:
        raise ValueError("No tanimoto_to_seed columns found in trajectories")

    rounds = [int(c.split("_")[1]) for c in tan_cols]

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df.iterrows():
        y_values = [row.get(c) for c in tan_cols]
        valid = [(r, v) for r, v in zip(rounds, y_values) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if valid:
            x, y = zip(*valid)
            ax.plot(x, y, "-o", alpha=0.3, markersize=3, linewidth=1, color="steelblue")

    # Mean drift
    mean_drifts = []
    for c in tan_cols:
        vals = df[c].dropna()
        mean_drifts.append(vals.mean() if len(vals) > 0 else None)

    valid_means = [(r, m) for r, m in zip(rounds, mean_drifts) if m is not None]
    if valid_means:
        mx, my = zip(*valid_means)
        ax.plot(mx, my, "o-", color="red", linewidth=2.5, markersize=6, label="Mean drift", zorder=5)

    ax.set_xlabel("Round")
    ax.set_ylabel("Tanimoto Distance from Initial Seed")
    ax.set_title(title or "Tanimoto Drift from Initial Seed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Best-of-N Distribution
# ---------------------------------------------------------------------------

def plot_best_of_n_distribution(
    results_df: pd.DataFrame,
    property_name: str,
    max_seeds: int = 12,
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Histogram of all N generations per seed, with the selected molecule highlighted.

    Parameters
    ----------
    results_df : pd.DataFrame
        ``all_generations.parquet`` from a best-of-N run (round=0).
        Must have columns ``seed_idx``, ``{property_name}``, ``selected``,
        ``is_valid``.
    property_name : str
        Property to histogram.
    max_seeds : int
        Maximum number of seeds to show (subplots).
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    df = results_df[results_df["is_valid"] == True].copy()  # noqa: E712
    seed_ids = sorted(df["seed_idx"].unique())[:max_seeds]

    ncols = min(4, len(seed_ids))
    nrows = (len(seed_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, seed_id in enumerate(seed_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        sdf = df[df["seed_idx"] == seed_id]
        values = sdf[property_name].dropna()
        selected = sdf[sdf["selected"] == True][property_name].dropna()  # noqa: E712

        if len(values) > 0:
            ax.hist(values, bins=20, alpha=0.6, color="steelblue", edgecolor="white")
        if len(selected) > 0:
            for sv in selected:
                ax.axvline(sv, color="red", linewidth=2, linestyle="--", label="Selected")

        ax.set_title(f"Seed {seed_id}", fontsize=9)
        ax.set_xlabel(property_name, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for idx in range(len(seed_ids), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title or f"Best-of-N Distribution ({property_name})", fontsize=12)
    fig.tight_layout()

    return _save_and_return(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Property Pareto Front
# ---------------------------------------------------------------------------

def plot_property_pareto(
    results_df: pd.DataFrame,
    prop_x: str,
    prop_y: str,
    direction_x: str = "increasing",
    direction_y: str = "increasing",
    highlight: str = "selected",
    save_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Scatter plot of two properties with Pareto front highlighted.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns ``{prop_x}``, ``{prop_y}``, ``is_valid``,
        ``selected``.
    prop_x : str
        Property for x-axis.
    prop_y : str
        Property for y-axis.
    direction_x : str
        ``"increasing"`` or ``"decreasing"`` for x property.
    direction_y : str
        ``"increasing"`` or ``"decreasing"`` for y property.
    highlight : str
        ``"selected"``: highlight selected molecules.
        ``"vin"``: highlight V,I,N molecules.
    save_path : str | None
        Path to save the figure.
    title : str | None
        Plot title override.

    Returns
    -------
    plt.Figure
    """
    df = results_df[results_df["is_valid"] == True].dropna(subset=[prop_x, prop_y]).copy()  # noqa: E712

    fig, ax = plt.subplots(figsize=(8, 6))

    # All valid
    ax.scatter(
        df[prop_x], df[prop_y],
        alpha=0.2, s=15, color="gray", label="All valid"
    )

    # Highlighted
    if highlight == "selected":
        hdf = df[df["selected"] == True]  # noqa: E712
        label = "Selected"
    else:
        vin_col = f"is_vin_{prop_x}"
        if vin_col in df.columns:
            hdf = df[df[vin_col] == True]  # noqa: E712
            label = "V,I,N"
        else:
            hdf = df[df["selected"] == True]  # noqa: E712
            label = "Selected"

    if len(hdf) > 0:
        ax.scatter(
            hdf[prop_x], hdf[prop_y],
            alpha=0.8, s=40, color="red", label=label, zorder=5
        )

    # Compute and plot Pareto front
    pareto = _compute_pareto_front(
        df[[prop_x, prop_y]].values,
        maximize_x=(direction_x == "increasing"),
        maximize_y=(direction_y == "increasing"),
    )
    if len(pareto) > 1:
        pareto_sorted = pareto[pareto[:, 0].argsort()]
        ax.plot(
            pareto_sorted[:, 0], pareto_sorted[:, 1],
            "k--", linewidth=1.5, alpha=0.7, label="Pareto front"
        )
        ax.scatter(
            pareto_sorted[:, 0], pareto_sorted[:, 1],
            marker="D", s=30, color="gold", edgecolors="black", zorder=6
        )

    ax.set_xlabel(prop_x)
    ax.set_ylabel(prop_y)
    ax.set_title(title or f"Pareto Front: {prop_x} vs {prop_y}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_and_return(fig, save_path)


def _compute_pareto_front(
    points: np.ndarray,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """Compute the Pareto front from a set of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2).
    maximize_x : bool
        If True, higher x is better.
    maximize_y : bool
        If True, higher y is better.

    Returns
    -------
    np.ndarray
        Points on the Pareto front.
    """
    # Negate dimensions we want to maximize (so we can use "minimize" logic)
    adjusted = points.copy()
    if maximize_x:
        adjusted[:, 0] = -adjusted[:, 0]
    if maximize_y:
        adjusted[:, 1] = -adjusted[:, 1]

    is_pareto = np.ones(len(adjusted), dtype=bool)
    for i in range(len(adjusted)):
        if not is_pareto[i]:
            continue
        # A point is dominated if another point is <= in all dimensions
        dominated = np.all(adjusted[is_pareto] <= adjusted[i], axis=1)
        # But not by itself
        strict = np.any(adjusted[is_pareto] < adjusted[i], axis=1)
        dominated = dominated & strict
        # Remove dominated points
        pareto_indices = np.where(is_pareto)[0]
        is_pareto[pareto_indices[dominated]] = False
        is_pareto[i] = True  # keep current point

    return points[is_pareto]
