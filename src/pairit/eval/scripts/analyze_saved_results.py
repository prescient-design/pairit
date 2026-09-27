#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Root of the pairit_eval results tree. Overridable via env var for public users
# who ship their own runs outside the packaged assets/.
REPO_ROOT = Path(
    os.environ.get("PAIRIT_EVAL_RESULTS_ROOT", "./pairit_eval_results")
).resolve()
PIPELINE_ROOT = REPO_ROOT / "results" / "pipeline"
OUT_DIR = PIPELINE_ROOT / "analysis"

DATASETS = ["bace", "bbbp", "hiv"]
METHODS = [
    "nfbo",
    "ddsbm-bridge",
    "ddsbm-sb",
    "mmp",
    "mmp-prop",
    # "hierg2g-04",  # kept for reference/support, but excluded from default comparisons
    "hierg2g-06",
    "pairit-from-zinc",
    "pairit-real-only",
]
ANALYSIS_METRICS = ["valid", "relevant", "improved", "sim_seed", "internal_diversity"]
PCT_METRICS = ["valid", "relevant", "improved"]
MC_SAMPLES_AVG = 200_000
MC_SAMPLES_SEED = 50_000


def scale_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in PCT_METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce") / 100.0
    for metric in set(ANALYSIS_METRICS) - set(PCT_METRICS):
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out


def hv_mc(points: np.ndarray, n_samples: int, seed: int) -> float:
    if len(points) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.random((n_samples, points.shape[1]))
    dominated = np.any(np.all(points[:, None, :] >= samples[None, :, :], axis=2), axis=0)
    return float(dominated.mean())


def pareto_front_methods(df: pd.DataFrame) -> list[str]:
    vals = df[ANALYSIS_METRICS].to_numpy(dtype=float)
    methods = df["method"].tolist()
    front: list[str] = []
    for i, method in enumerate(methods):
        dominated = False
        for j, _ in enumerate(methods):
            if i == j:
                continue
            if (vals[j] >= vals[i]).all() and (vals[j] > vals[i]).any():
                dominated = True
                break
        if not dominated:
            front.append(method)
    return front


def load_average_results() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        path = PIPELINE_ROOT / method / f"{method}_average.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "method" not in df.columns:
            df["method"] = method
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No average files found under {PIPELINE_ROOT}")
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out["method"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out = out[out["dataset"].isin(DATASETS)].copy()
    return out


def load_per_seed_results() -> pd.DataFrame:
    frames = []
    for method in METHODS:
        path = PIPELINE_ROOT / method / f"{method}_per_seed.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "method" not in df.columns:
            df["method"] = method
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No per-seed files found under {PIPELINE_ROOT}")
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out["method"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out = out[out["dataset"].isin(DATASETS)].copy()
    return out


def choose_representative_pairit(avg_scaled: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    pairit = avg_scaled[avg_scaled["method"].str.startswith("pairit")].copy()
    rank_rows = []
    for dataset in DATASETS:
        sub = pairit[pairit["dataset"] == dataset].dropna(subset=ANALYSIS_METRICS).copy()
        if len(sub) == 0:
            continue
        for metric in ANALYSIS_METRICS:
            sub[f"{metric}_rank"] = sub[metric].rank(ascending=False, method="min")
        sub["rank_sum"] = sub[[f"{metric}_rank" for metric in ANALYSIS_METRICS]].sum(axis=1)
        rank_rows.append(sub[["method", "dataset", "rank_sum"] + [f"{metric}_rank" for metric in ANALYSIS_METRICS]])
    rank_df = pd.concat(rank_rows, ignore_index=True)
    choice_df = (
        rank_df.groupby("method", observed=True, as_index=False)["rank_sum"]
        .mean()
        .rename(columns={"rank_sum": "mean_rank_sum"})
        .sort_values(["mean_rank_sum", "method"])
        .reset_index(drop=True)
    )
    representative = str(choice_df.iloc[0]["method"])
    return representative, rank_df, choice_df


def build_method_view(avg_scaled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = avg_scaled[avg_scaled["dataset"] == dataset].dropna(subset=ANALYSIS_METRICS).copy()
        if len(sub) == 0:
            continue
        front = set(pareto_front_methods(sub))
        sub["single_point_hv"] = sub[ANALYSIS_METRICS].prod(axis=1)
        sub["on_pareto_front"] = sub["method"].isin(front)
        rows.append(sub[["method", "dataset"] + ANALYSIS_METRICS + ["single_point_hv", "on_pareto_front"]])
    return pd.concat(rows, ignore_index=True)


def compare_vs_nfbo(method_view: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = method_view[method_view["dataset"] == dataset].copy()
        if "nfbo" not in set(sub["method"]):
            continue
        ref = sub[sub["method"] == "nfbo"].iloc[0]
        for _, row in sub.iterrows():
            rec = {
                "dataset": dataset,
                "method": row["method"],
                "on_pareto_front": row["on_pareto_front"],
                "single_point_hv": row["single_point_hv"],
                "delta_hv_vs_nfbo": row["single_point_hv"] - ref["single_point_hv"],
            }
            for metric in ANALYSIS_METRICS:
                rec[f"delta_{metric}_vs_nfbo"] = row[metric] - ref[metric]
            rows.append(rec)
    return pd.DataFrame(rows)


def set_gain_summary(avg_scaled: pd.DataFrame, per_seed_scaled: pd.DataFrame, representative: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_rows = []
    seed_rows = []
    for dataset in DATASETS:
        avg_sub = avg_scaled[avg_scaled["dataset"] == dataset].dropna(subset=ANALYSIS_METRICS).copy()
        seed_sub = per_seed_scaled[per_seed_scaled["dataset"] == dataset].copy()
        for exclude_nfbo in (False, True):
            if exclude_nfbo:
                avg_base = avg_sub[(~avg_sub["method"].str.startswith("pairit")) & (avg_sub["method"] != "nfbo")].copy()
            else:
                avg_base = avg_sub[(~avg_sub["method"].str.startswith("pairit"))].copy()
            avg_pair = avg_sub[avg_sub["method"] == representative].copy()
            if len(avg_pair) == 1 and len(avg_base):
                hv_base = hv_mc(avg_base[ANALYSIS_METRICS].to_numpy(), MC_SAMPLES_AVG, seed=1)
                hv_plus = hv_mc(pd.concat([avg_base, avg_pair], ignore_index=True)[ANALYSIS_METRICS].to_numpy(), MC_SAMPLES_AVG, seed=1)
                avg_rows.append({
                    "dataset": dataset,
                    "comparison": "w/o nfbo" if exclude_nfbo else "all baselines",
                    "hv_baselines": hv_base,
                    "hv_with_pairit": hv_plus,
                    "hv_gain": hv_plus - hv_base,
                    "pairit_on_pareto_front": representative in pareto_front_methods(pd.concat([avg_base, avg_pair], ignore_index=True)),
                })

            gains = []
            fronts = []
            n_complete = 0
            expected_methods = 6 if exclude_nfbo else 7
            for seed_idx, grp in seed_sub.groupby("seed_idx"):
                if exclude_nfbo:
                    comp = grp[((~grp["method"].str.startswith("pairit")) & (grp["method"] != "nfbo")) | (grp["method"] == representative)].copy()
                else:
                    comp = grp[((~grp["method"].str.startswith("pairit")) | (grp["method"] == representative))].copy()
                if comp["method"].nunique() != expected_methods:
                    continue
                if comp[ANALYSIS_METRICS].isna().any().any():
                    continue
                base = comp[comp["method"] != representative]
                hv_base = hv_mc(base[ANALYSIS_METRICS].to_numpy(), MC_SAMPLES_SEED, seed=int(seed_idx) + (111 if exclude_nfbo else 11))
                hv_plus = hv_mc(comp[ANALYSIS_METRICS].to_numpy(), MC_SAMPLES_SEED, seed=int(seed_idx) + (111 if exclude_nfbo else 11))
                gains.append(hv_plus - hv_base)
                fronts.append(representative in pareto_front_methods(comp))
                n_complete += 1
            gains_arr = np.asarray(gains, dtype=float)
            seed_rows.append({
                "dataset": dataset,
                "comparison": "w/o nfbo" if exclude_nfbo else "all baselines",
                "n_complete_seeds": n_complete,
                "seed_front_rate": float(np.mean(fronts)) if fronts else np.nan,
                "seed_hv_gain_mean": float(gains_arr.mean()) if len(gains_arr) else np.nan,
                "seed_hv_gain_std": float(gains_arr.std(ddof=1)) if len(gains_arr) > 1 else np.nan,
                "seed_hv_gain_median": float(np.median(gains_arr)) if len(gains_arr) else np.nan,
                "seed_positive_gain_rate": float(np.mean(gains_arr > 0)) if len(gains_arr) else np.nan,
            })
    return pd.DataFrame(avg_rows), pd.DataFrame(seed_rows)


def write_summary_markdown(
    out_path: Path,
    representative: str,
    choice_df: pd.DataFrame,
    method_view: pd.DataFrame,
    compare_df: pd.DataFrame,
    avg_gain_df: pd.DataFrame,
    seed_gain_df: pd.DataFrame,
) -> None:
    best_rows = []
    rep_rows = method_view[method_view["method"] == representative].copy()
    nfbo_rows = method_view[method_view["method"] == "nfbo"].copy()
    for dataset in DATASETS:
        sub = method_view[method_view["dataset"] == dataset].copy()
        if len(sub) == 0:
            continue
        best = sub.sort_values("single_point_hv", ascending=False).iloc[0]
        rep = rep_rows[rep_rows["dataset"] == dataset].iloc[0]
        nfbo = nfbo_rows[nfbo_rows["dataset"] == dataset].iloc[0] if len(nfbo_rows[nfbo_rows["dataset"] == dataset]) else None
        rep_vs_nfbo = compare_df[(compare_df["dataset"] == dataset) & (compare_df["method"] == representative)]
        rep_vs_nfbo = rep_vs_nfbo.iloc[0] if len(rep_vs_nfbo) else None
        avg_all = avg_gain_df[(avg_gain_df["dataset"] == dataset) & (avg_gain_df["comparison"] == "all baselines")].iloc[0]
        avg_wo = avg_gain_df[(avg_gain_df["dataset"] == dataset) & (avg_gain_df["comparison"] == "w/o nfbo")].iloc[0]
        seed_all = seed_gain_df[(seed_gain_df["dataset"] == dataset) & (seed_gain_df["comparison"] == "all baselines")].iloc[0]
        seed_wo = seed_gain_df[(seed_gain_df["dataset"] == dataset) & (seed_gain_df["comparison"] == "w/o nfbo")].iloc[0]
        meaning = []
        if nfbo is not None:
            if rep["single_point_hv"] > nfbo["single_point_hv"]:
                meaning.append("representative PairIt has higher single-point HV than nfbo")
            else:
                meaning.append("representative PairIt trails nfbo on single-point HV")
            if rep_vs_nfbo is not None:
                if rep_vs_nfbo["delta_improved_vs_nfbo"] < 0:
                    meaning.append("main weakness vs nfbo is lower property improvement")
                if rep_vs_nfbo["delta_valid_vs_nfbo"] > 0:
                    meaning.append("PairIt is stronger on validity")
                if rep_vs_nfbo["delta_relevant_vs_nfbo"] > 0:
                    meaning.append("PairIt is slightly stronger on relevance")
        if bool(rep["on_pareto_front"]):
            meaning.append("PairIt is on the Pareto front")
        best_rows.append(
            f"### {dataset}\n"
            f"- Best single-point HV method: `{best['method']}` ({best['single_point_hv']:.4f})\n"
            f"- Representative PairIt: `{representative}` ({rep['single_point_hv']:.4f})\n"
            f"- Avg-level HV gain from adding PairIt to all baselines: `{avg_all['hv_gain']:.4f}`\n"
            f"- Avg-level HV gain without nfbo: `{avg_wo['hv_gain']:.4f}`\n"
            f"- Seed-level front rate vs all baselines: `{seed_all['seed_front_rate']:.3f}` over `{int(seed_all['n_complete_seeds'])}` complete seeds\n"
            f"- Seed-level front rate without nfbo: `{seed_wo['seed_front_rate']:.3f}` over `{int(seed_wo['n_complete_seeds'])}` complete seeds\n"
            f"- Meaning: " + "; ".join(meaning) + "\n"
        )

    text = (
        "# Saved Results Analysis\n\n"
        f"Representative PairIt method chosen from average cross-dataset ranks: `{representative}`.\n\n"
        "## PairIt Choice\n\n"
        + choice_df.to_markdown(index=False)
        + "\n\n## Dataset Summaries\n\n"
        + "\n".join(best_rows)
    )
    out_path.write_text(text, encoding="utf-8")


def plot_method_hv(method_view: pd.DataFrame, representative: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.2 * len(DATASETS), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, DATASETS):
        sub = method_view[method_view["dataset"] == dataset].sort_values("single_point_hv", ascending=False).copy()
        positions = np.arange(len(sub))
        colors = []
        for method in sub["method"]:
            if method.startswith("pairit"):
                colors.append("#F58518" if method == representative else "#FFB55A")
            elif method == "nfbo":
                colors.append("#54A24B")
            else:
                colors.append("#4C78A8")
        bars = ax.bar(positions, sub["single_point_hv"], color=colors)
        for bar, on_front in zip(bars, sub["on_pareto_front"]):
            if on_front:
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.8)
            else:
                bar.set_edgecolor("#111111")
                bar.set_linewidth(2.0)
                bar.set_hatch("//")
                bar.set_alpha(0.7)
        ax.set_title(dataset)
        ax.set_ylabel("Single-point HV")
        ax.set_xticks(positions)
        ax.set_xticklabels(sub["method"], rotation=35, ha="right")
    fig.suptitle("Method-by-Method Hypervolume (hatched = dominated / off Pareto front)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tradeoff_scatter(method_view: pd.DataFrame, representative: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.4 * len(DATASETS), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    valid_min = float(method_view["valid"].min())
    valid_max = float(method_view["valid"].max())
    valid_span = max(valid_max - valid_min, 1e-8)
    short_labels = {
        "pairit-from-zinc": "pairit-zinc",
        "pairit-real-only": "pairit-real",
        "ddsbm-bridge": "ddsbm-br",
        "ddsbm-sb": "ddsbm-sb",
        "hierg2g-04": "hierg2g-04",
        "hierg2g-06": "hierg2g-06",
        "nfbo": "nfbo",
        "mmp": "mmp",
        "mmp-prop": "mmp-prop",
    }
    label_offsets = [(8, 8), (8, -10), (-10, 8), (-10, -10), (12, 0), (-12, 0), (0, 12), (0, -12)]
    for ax, dataset in zip(axes, DATASETS):
        sub = method_view[method_view["dataset"] == dataset].copy()
        edgecolors = ["black" if x else "none" for x in sub["on_pareto_front"]]
        valid_scaled = (sub["valid"] - valid_min) / valid_span
        marker_sizes = 250 + 3250 * np.power(valid_scaled, 1.35)
        scatter = ax.scatter(
            sub["sim_seed"],
            sub["improved"],
            s=marker_sizes,
            c=sub["relevant"],
            cmap="viridis",
            edgecolors=edgecolors,
            linewidths=2,
            alpha=0.78,
        )
        sub = sub.sort_values(["improved", "sim_seed"], ascending=[False, False]).reset_index(drop=True)
        for idx, row in sub.iterrows():
            label = short_labels.get(row["method"], row["method"])
            if row["method"] == representative:
                label = f"{label} *"
            dx, dy = label_offsets[idx % len(label_offsets)]
            ax.annotate(
                label,
                xy=(row["sim_seed"], row["improved"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
            )
        ax.set_title(dataset)
        ax.set_xlabel("sim_seed")
        ax.set_ylabel("improved")
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("relevant")
    fig.suptitle("Improved vs sim_seed (size = valid, color = relevant, outline = Pareto front)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pairit_vs_nfbo(compare_df: pd.DataFrame, representative: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.4 * len(DATASETS), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    metric_labels = ["valid", "relevant", "improved", "sim_seed", "internal_diversity"]
    delta_cols = [f"delta_{m}_vs_nfbo" for m in metric_labels]
    for ax, dataset in zip(axes, DATASETS):
        row = compare_df[(compare_df["dataset"] == dataset) & (compare_df["method"] == representative)].iloc[0]
        vals = [row[col] for col in delta_cols]
        colors = ["#54A24B" if v >= 0 else "#E45756" for v in vals]
        ax.bar(metric_labels, vals, color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(dataset)
        ax.set_ylabel(f"{representative} - nfbo")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Representative PairIt vs nfbo by metric")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_set_gain(avg_gain_df: pd.DataFrame, seed_gain_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = 0.38
    x = np.arange(len(DATASETS))
    avg_all = [avg_gain_df[(avg_gain_df["dataset"] == ds) & (avg_gain_df["comparison"] == "all baselines")]["hv_gain"].iloc[0] for ds in DATASETS]
    avg_wo = [avg_gain_df[(avg_gain_df["dataset"] == ds) & (avg_gain_df["comparison"] == "w/o nfbo")]["hv_gain"].iloc[0] for ds in DATASETS]
    axes[0].bar(x - width / 2, avg_all, width=width, label="all baselines", color="#4C78A8")
    axes[0].bar(x + width / 2, avg_wo, width=width, label="w/o nfbo", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(DATASETS)
    axes[0].set_ylabel("HV gain")
    axes[0].set_title("Average-level set HV gain from adding representative PairIt")
    axes[0].legend()

    seed_all = [seed_gain_df[(seed_gain_df["dataset"] == ds) & (seed_gain_df["comparison"] == "all baselines")]["seed_front_rate"].iloc[0] for ds in DATASETS]
    seed_wo = [seed_gain_df[(seed_gain_df["dataset"] == ds) & (seed_gain_df["comparison"] == "w/o nfbo")]["seed_front_rate"].iloc[0] for ds in DATASETS]
    axes[1].bar(x - width / 2, seed_all, width=width, label="all baselines", color="#4C78A8")
    axes[1].bar(x + width / 2, seed_wo, width=width, label="w/o nfbo", color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(DATASETS)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Front rate")
    axes[1].set_title("Seed-level Pareto front rate for representative PairIt")
    axes[1].legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    avg_raw = load_average_results()
    per_seed_raw = load_per_seed_results()
    avg_scaled = scale_metrics(avg_raw[["method", "dataset"] + ANALYSIS_METRICS].copy()).dropna(subset=ANALYSIS_METRICS)
    per_seed_scaled = scale_metrics(per_seed_raw[["method", "dataset", "seed_idx"] + ANALYSIS_METRICS].copy())

    representative, rank_df, choice_df = choose_representative_pairit(avg_scaled)
    method_view = build_method_view(avg_scaled)
    compare_df = compare_vs_nfbo(method_view)
    avg_gain_df, seed_gain_df = set_gain_summary(avg_scaled, per_seed_scaled, representative)

    avg_raw.to_csv(OUT_DIR / "average_metrics_raw.csv", index=False)
    rank_df.to_csv(OUT_DIR / "pairit_rank_selection.csv", index=False)
    choice_df.to_csv(OUT_DIR / "pairit_choice_summary.csv", index=False)
    method_view.sort_values(["dataset", "single_point_hv"], ascending=[True, False]).to_csv(
        OUT_DIR / "method_view_hv_pareto.csv", index=False
    )
    compare_df.sort_values(["dataset", "delta_hv_vs_nfbo"], ascending=[True, False]).to_csv(
        OUT_DIR / "representative_pairit_vs_nfbo.csv", index=False
    )
    avg_gain_df.to_csv(OUT_DIR / "representative_pairit_set_gain_average.csv", index=False)
    seed_gain_df.to_csv(OUT_DIR / "representative_pairit_set_gain_seed.csv", index=False)

    write_summary_markdown(
        OUT_DIR / "summary.md",
        representative,
        choice_df,
        method_view,
        compare_df,
        avg_gain_df,
        seed_gain_df,
    )

    plot_method_hv(method_view, representative, OUT_DIR / "method_hv_by_dataset.png")
    plot_tradeoff_scatter(method_view, representative, OUT_DIR / "tradeoff_scatter_by_dataset.png")
    plot_pairit_vs_nfbo(compare_df, representative, OUT_DIR / "representative_pairit_vs_nfbo.png")
    plot_set_gain(avg_gain_df, seed_gain_df, OUT_DIR / "representative_pairit_set_gain.png")

    print(f"Saved analysis to: {OUT_DIR}")
    print(f"Representative PairIt: {representative}")
    print("\nMethod-by-method HV summary:")
    print(method_view.sort_values(["dataset", "single_point_hv"], ascending=[True, False]).to_string(index=False))
    print("\nRepresentative PairIt vs nfbo:")
    print(compare_df.sort_values(["dataset", "delta_hv_vs_nfbo"], ascending=[True, False]).to_string(index=False))
    print("\nAverage-level set gain summary:")
    print(avg_gain_df.to_string(index=False))
    print("\nSeed-level set gain summary:")
    print(seed_gain_df.to_string(index=False))


if __name__ == "__main__":
    main()
