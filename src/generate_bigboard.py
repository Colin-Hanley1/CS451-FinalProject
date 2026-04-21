from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA

from src.config import BIGBOARD_CSV, MODELS_DIR, SITE_DATA_DIR
from src.preprocess import (compute_position_stats, engineer_features,
                            feature_columns, guard_ws40, load_clean_data,
                            three_pt_shrinkage_params)

def load_artifacts():
    return {
        "scaler":  joblib.load(MODELS_DIR / "scaler.pkl"),
        "xgb_reg": joblib.load(MODELS_DIR / "xgboost_reg.pkl"),
        "xgb_clf": joblib.load(MODELS_DIR / "xgboost_clf.pkl"),
        "kmeans":  joblib.load(MODELS_DIR / "kmeans.pkl"),
        "meta":    json.loads((SITE_DATA_DIR / "model_meta.json").read_text()),
    }

def predict_clusters(kmeans, bb: pd.DataFrame, X_bb_s: np.ndarray) -> np.ndarray:
    cluster = np.full(len(bb), -1, dtype=int)
    for pos, entry in kmeans.items():
        mask = (bb["position"].values == pos)
        if mask.any():
            local = entry["model"].predict(X_bb_s[mask])
            cluster[mask] = local + entry["offset"]
    return cluster

def main():
    if not BIGBOARD_CSV.exists():
        raise SystemExit(f"2026class.csv not found at {BIGBOARD_CSV}")

    art = load_artifacts()
    scaler, xgb_reg, xgb_clf = art["scaler"], art["xgb_reg"], art["xgb_clf"]
    feature_cols = art["meta"]["feature_names"]

    df_clean = load_clean_data()
    pos_stats = compute_position_stats(df_clean)
    league_3p, k_3p = three_pt_shrinkage_params(df_clean)
    df_clean_feat = engineer_features(df_clean, pos_stats, league_3p, k_3p)
    if "three_pt_pct" in df_clean_feat.columns:
        df_clean_feat = df_clean_feat.drop(columns=["three_pt_pct"])
    X_all_s = scaler.transform(df_clean_feat[feature_cols].values)

    bb_raw = pd.read_csv(BIGBOARD_CSV)
    bb_raw["draft_year"] = 2026
    bb_raw = guard_ws40(bb_raw, training_max=float(df_clean["ws_40"].max()))
    bb = engineer_features(bb_raw, pos_stats, league_3p, k_3p)
    if "three_pt_pct" in bb.columns:

        bb_raw_3p = bb_raw["three_pt_pct"].values
    else:
        bb_raw_3p = np.full(len(bb), np.nan)

    X_bb = bb[feature_cols].values
    X_bb_s = scaler.transform(X_bb)

    pred_vorp = xgb_reg.predict(X_bb_s)
    pred_prob = xgb_clf.predict_proba(X_bb_s)[:, 1]
    cluster = predict_clusters(art["kmeans"], bb, X_bb_s)

    pca = PCA(n_components=2).fit(X_all_s)
    bb_pca = pca.transform(X_bb_s)

    explainer = shap.TreeExplainer(xgb_reg)
    bb_shap = explainer.shap_values(X_bb_s)

    df_position = df_clean_feat["position"].values
    rows = []
    for i, (_, r) in enumerate(bb.iterrows()):
        shap_pairs = sorted(
            [(feature_cols[j], float(bb_shap[i][j])) for j in range(len(feature_cols))],
            key=lambda x: abs(x[1]), reverse=True,
        )[:5]

        dists = np.linalg.norm(X_all_s - X_bb_s[i], axis=1)
        dists = np.where(df_position == r["position"], dists, np.inf)
        comp_idx = np.argsort(dists)[:5]
        comps = [{
            "player_name": df_clean_feat.iloc[ci]["player_name"],
            "draft_year":  int(df_clean_feat.iloc[ci]["draft_year"]),
            "college":     df_clean_feat.iloc[ci]["college"],
            "position":    df_clean_feat.iloc[ci]["position"],
            "actual_vorp": round(float(df_clean_feat.iloc[ci]["nba_4yr_vorp"]), 1),
        } for ci in comp_idx]

        rows.append({
            "player_name":           r["player_name"],
            "college":               r["college"],
            "position":              r["position"],
            "height_inches":         int(r["height_inches"]),
            "weight_lbs":            int(r["weight_lbs"]),
            "ppg":                   round(float(r["ppg"]), 1),
            "apg":                   round(float(r["apg"]), 1),
            "rpg":                   round(float(r["rpg"]), 1),
            "spg":                   round(float(r["spg"]), 2),
            "bpg":                   round(float(r["bpg"]), 2),
            "ts_pct":                round(float(r["ts_pct"]), 3),
            "ws_40":                 round(float(r["ws_40"]), 3),
            "three_pt_pct":          round(float(bb_raw_3p[i]), 3) if not np.isnan(bb_raw_3p[i]) else 0.0,
            "three_pt_attempt_rate": round(float(r["three_pt_attempt_rate"]), 3),
            "recruit_rank":          int(r["recruit_rank"]),
            "college_seasons":       int(r["college_seasons"]),
            "predicted_vorp":        round(float(pred_vorp[i]), 2),
            "posvorp_prob":          round(float(pred_prob[i]), 3),
            "cluster":               int(cluster[i]),
            "pca_1":                 round(float(bb_pca[i, 0]), 3),
            "pca_2":                 round(float(bb_pca[i, 1]), 3),
            "shap_features":         [{"feature": f, "shap": round(s, 3)} for f, s in shap_pairs],
            "comps":                 comps,
        })

    rows.sort(key=lambda x: -x["predicted_vorp"])
    for rank, p in enumerate(rows, start=1):
        p["rank"] = rank

    out = SITE_DATA_DIR / "bigboard.json"
    out.write_text(json.dumps(rows))
    print(f"Wrote {len(rows)} rows to {out}")
    print("Top 10 by projected VORP:")
    for p in rows[:10]:
        print(f"  {p['rank']:>2}. {p['player_name']:25s} ({p['college']:20s}) "
              f"-> {p['predicted_vorp']:+.2f}  P(VORP+)={p['posvorp_prob']:.2f}")

if __name__ == "__main__":
    main()
