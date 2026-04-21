from __future__ import annotations

import pandas as pd

from src.config import DATA_CSV, POSITION_REL_STATS

def fix_known_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["player_name"].str.contains("N'Diaye", na=False), "fg_pct"] = 0.539
    df.loc[df["player_name"].str.contains("Tristan Thompson", na=False), "fg_pct"] = 0.546
    return df

def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = fix_known_bad_rows(df)
    return df.dropna(subset=["nba_4yr_vorp"]).copy()

def compute_position_stats(df_clean: pd.DataFrame) -> dict:
    stats = list(POSITION_REL_STATS) + ["height_inches", "weight_lbs"]
    return {
        stat: {
            p: {
                "mean": float(df_clean[df_clean["position"] == p][stat].mean()),
                "std":  float(df_clean[df_clean["position"] == p][stat].std()),
            }
            for p in df_clean["position"].unique()
        }
        for stat in stats
    }

def engineer_features(
    df: pd.DataFrame,
    pos_stats: dict,
    league_3p_mean: float,
    three_pt_shrinkage_k: float,
) -> pd.DataFrame:
    out = df.copy()

    out["recruit_rank"] = pd.to_numeric(
        out["recruit_rank"], errors="coerce"
    ).fillna(101).astype(int)

    if "is_top_100_recruit" not in out.columns:
        out["is_top_100_recruit"] = (out["recruit_rank"] <= 100).astype(int)
    if "is_top_10_recruit" not in out.columns:
        out["is_top_10_recruit"] = (out["recruit_rank"] <= 10).astype(int)
    if "one_and_done" not in out.columns:
        out["one_and_done"] = (out["college_seasons"] == 1).astype(int)

    for stat in POSITION_REL_STATS:
        ms = pos_stats[stat]
        out[f"{stat}_vs_pos"] = out.apply(
            lambda r: (r[stat] - ms[r["position"]]["mean"]) / ms[r["position"]]["std"]
            if ms[r["position"]]["std"] > 0 else 0,
            axis=1,
        )

    out["ws40_per_season"] = out["ws_40"] / out["college_seasons"].clip(lower=1)

    out["def_versatility"] = out["spg"] * out["bpg"]

    out["recruit_x_ws40"] = (101 - out["recruit_rank"]) * out["ws_40"]

    out["three_pt_pct_vw"] = (
        (out["three_pt_pct"] * out["three_pt_attempt_rate"]
         + league_3p_mean * three_pt_shrinkage_k)
        / (out["three_pt_attempt_rate"] + three_pt_shrinkage_k)
    )

    for stat in ["height_inches", "weight_lbs"]:
        ms = pos_stats[stat]
        out[f"{stat}_vs_pos"] = out.apply(
            lambda r: (r[stat] - ms[r["position"]]["mean"]) / ms[r["position"]]["std"]
            if ms[r["position"]]["std"] > 0 else 0,
            axis=1,
        )
    out["build_score"] = out["height_inches_vs_pos"] + out["weight_lbs_vs_pos"]

    return out

def three_pt_shrinkage_params(df_clean: pd.DataFrame) -> tuple[float, float]:
    return (
        float(df_clean["three_pt_pct"].mean()),
        float(df_clean["three_pt_attempt_rate"].quantile(0.25)),
    )

def guard_ws40(df: pd.DataFrame, training_max: float) -> pd.DataFrame:
    out = df.copy()
    bad = out["ws_40"] > training_max * 1.2
    if bad.any():
        est = out["ws"] / (out["mpg"] * 30 * out["college_seasons"]).clip(lower=1) * 40
        for idx in out[bad].index:
            print(f"  capping ws_40 for {out.loc[idx, 'player_name']}: "
                  f"{out.loc[idx, 'ws_40']:.3f} -> {est.loc[idx]:.3f}")
        out.loc[bad, "ws_40"] = est[bad]
    return out

def feature_columns(df: pd.DataFrame) -> list[str]:
    from src.config import IDENTITY_COLS, TARGET_COLS
    excluded = set(IDENTITY_COLS + TARGET_COLS + ["position"])
    return [c for c in df.columns if c not in excluded]
