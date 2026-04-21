from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_CSV, FIGURES_DIR

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")

def fig_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df["nba_4yr_vorp"].dropna().hist(bins=40, ax=axes[0], color="#2c5f8a", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7, label="VORP = 0")
    axes[0].axvline(df["nba_4yr_vorp"].dropna().median(), color="orange",
                    linestyle="--", alpha=0.7,
                    label=f"Median = {df['nba_4yr_vorp'].dropna().median():.1f}")
    axes[0].set_xlabel("4-Year VORP")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of NBA 4-Year VORP")
    axes[0].legend()

    df["nba_4yr_bpm"].dropna().hist(bins=40, ax=axes[1], color="#5a8a3c", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("4-Year BPM")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of NBA 4-Year BPM")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close()

def fig_pick_vs_vorp(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = df["nba_4yr_vorp"].notna()
    ax.scatter(df.loc[mask, "pick_number"], df.loc[mask, "nba_4yr_vorp"],
               alpha=0.35, s=20, c="#2c5f8a")
    z = np.polyfit(df.loc[mask, "pick_number"], df.loc[mask, "nba_4yr_vorp"], 2)
    x_line = np.linspace(1, 60, 100)
    ax.plot(x_line, np.polyval(z, x_line), color="red", linewidth=2, label="Quadratic trend")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Draft Pick Number")
    ax.set_ylabel("4-Year VORP")
    ax.set_title("Draft Position vs. NBA Early Career VORP")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_pick_vs_vorp.png", bbox_inches="tight")
    plt.close()

def fig_vorp_by_position(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pos_order = ["Guard", "Forward", "Center"]
    colors = ["#2c5f8a", "#5a8a3c", "#c0392b"]
    data_by_pos = [df.loc[df["position"] == p, "nba_4yr_vorp"].dropna() for p in pos_order]
    bp = ax.boxplot(data_by_pos, labels=pos_order, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.3))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("4-Year VORP")
    ax.set_title("NBA 4-Year VORP by Position")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_position_analysis.png", bbox_inches="tight")
    plt.close()

def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    corr_cols = ["ppg", "apg", "rpg", "spg", "bpg", "mpg", "ts_pct", "fg_pct", "ft_pct",
                 "ws_40", "tov_pct", "ftr", "three_pt_attempt_rate", "recruit_rank",
                 "college_seasons", "height_inches", "weight_lbs", "nba_4yr_vorp"]
    corr_matrix = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-0.5, vmax=0.5,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (with 4-Year VORP)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

def fig_recruiting_impact(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_tmp = df.dropna(subset=["nba_4yr_vorp"]).copy()
    df_tmp["recruit_tier"] = pd.cut(df_tmp["recruit_rank"],
                                    bins=[-1, 10, 100, 102],
                                    labels=["Top 10", "Top 100", "Unranked"])
    tier_order = ["Top 10", "Top 100", "Unranked"]
    data_by_tier = [df_tmp.loc[df_tmp["recruit_tier"] == t, "nba_4yr_vorp"] for t in tier_order]
    bp = axes[0].boxplot(data_by_tier, labels=tier_order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#d4850a", "#2c5f8a", "#a0aec0"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("4-Year VORP")
    axes[0].set_title("VORP by Recruiting Tier")

    hit_rates = df_tmp.groupby("recruit_tier")["PosVORP"].mean() * 100
    hit_rates = hit_rates.reindex(tier_order)
    axes[1].bar(tier_order, hit_rates, color=["#d4850a", "#2c5f8a", "#a0aec0"],
                alpha=0.7, edgecolor="white")
    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(hit_rates):
        axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    axes[1].set_ylabel("% with Positive VORP")
    axes[1].set_title("Positive VORP Rate by Recruiting Tier")
    axes[1].set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_recruiting_impact.png", bbox_inches="tight")
    plt.close()

def fig_draft_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly = df.groupby("draft_year").agg(
        mean_vorp=("nba_4yr_vorp", "mean"),
        count=("nba_4yr_vorp", "count"),
    ).dropna()
    ax.bar(yearly.index, yearly["mean_vorp"], color="#2c5f8a", alpha=0.7, edgecolor="white")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Draft Year")
    ax.set_ylabel("Mean 4-Year VORP")
    ax.set_title("Average VORP by Draft Class")
    for yr, row in yearly.iterrows():
        ax.text(yr, row["mean_vorp"] + 0.15, f"n={row['count']:.0f}",
                ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_draft_trends.png", bbox_inches="tight")
    plt.close()

def fig_missing_data(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        plt.close()
        return
    missing.plot.barh(ax=ax, color="#c0392b", alpha=0.7)
    ax.set_xlabel("Number of Missing Values")
    ax.set_title("Missing Data by Column")
    for i, v in enumerate(missing):
        ax.text(v + 1, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_missing_data.png", bbox_inches="tight")
    plt.close()

def fig_feature_correlations(df_clean: pd.DataFrame, feature_cols: list) -> None:
    feat_corr = df_clean[feature_cols + ["nba_4yr_vorp"]].corr()["nba_4yr_vorp"].drop("nba_4yr_vorp")
    feat_corr = feat_corr.sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2d8a4e" if v > 0 else "#c0392b" for v in feat_corr.head(20)]
    feat_corr.head(20).plot.barh(ax=ax, color=colors, alpha=0.7)
    ax.set_xlabel("Correlation with 4-Year VORP")
    ax.set_title("Top 20 Feature Correlations with NBA VORP")
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_feature_correlations.png", bbox_inches="tight")
    plt.close()

def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} rows from {DATA_CSV.name}")

    fig_target_distribution(df)
    fig_pick_vs_vorp(df)
    fig_vorp_by_position(df)
    fig_correlation_heatmap(df)
    fig_recruiting_impact(df)
    fig_draft_trends(df)
    fig_missing_data(df)
    print("Wrote EDA figures 01-07 to figures/")

if __name__ == "__main__":
    main()
