from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.config import (K_BY_POSITION, MODELS_DIR, POSITION_ORDER, PROFILE_COLS,
                        RANDOM_STATE, SITE_DATA_DIR)
from src.preprocess import (compute_position_stats, engineer_features,
                            load_clean_data, three_pt_shrinkage_params)

def main():
    meta = json.loads((SITE_DATA_DIR / "model_meta.json").read_text())
    feature_cols = meta["feature_names"]
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    df = load_clean_data()
    pos_stats = compute_position_stats(df)
    league_3p, k_3p = three_pt_shrinkage_params(df)
    df = engineer_features(df, pos_stats, league_3p, k_3p)
    if "three_pt_pct" in df.columns:
        df = df.drop(columns=["three_pt_pct"])
    X_all_s = scaler.transform(df[feature_cols].values)

    pca = PCA(n_components=2).fit(X_all_s)
    pcs = pca.transform(X_all_s)

    km_models = {}
    global_cluster = np.full(len(df), -1, dtype=int)
    offset = 0
    for pos in POSITION_ORDER:
        k = K_BY_POSITION[pos]
        mask = (df["position"].values == pos)
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_all_s[mask])
        global_cluster[mask] = km.labels_ + offset
        km_models[pos] = {"model": km, "offset": offset, "k": k}
        print(f"  {pos:8s}: k={k}, n={mask.sum()}, inertia={km.inertia_:.1f}")
        offset += k

    df["cluster"] = global_cluster
    df["pca_1"] = pcs[:, 0]
    df["pca_2"] = pcs[:, 1]
    joblib.dump(km_models, MODELS_DIR / "kmeans.pkl")

    cluster_profiles = {}
    for pos in POSITION_ORDER:
        off = km_models[pos]["offset"]
        for local in range(km_models[pos]["k"]):
            gid = off + local
            mask = df["cluster"] == gid
            if mask.sum() == 0:
                continue
            profile = {col: round(float(df.loc[mask, col].mean()), 2) for col in PROFILE_COLS}
            profile["position"] = pos
            profile["local_id"] = local
            profile["count"] = int(mask.sum())
            profile["mean_vorp"] = round(float(df.loc[mask, "nba_4yr_vorp"].mean()), 2)
            cluster_profiles[str(gid)] = profile

    (SITE_DATA_DIR / "clusters.json").write_text(json.dumps({
        "n_clusters": sum(K_BY_POSITION.values()),
        "clustered_by_position": True,
        "k_per_position": K_BY_POSITION,
        "pca_variance_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_[:2]],
        "cluster_profiles": cluster_profiles,
    }, indent=2))
    print(f"\nWrote clusters.json ({sum(K_BY_POSITION.values())} archetypes)")

    prospects_path = SITE_DATA_DIR / "prospects.json"
    if prospects_path.exists():
        prospects = json.loads(prospects_path.read_text())
        lookup = {(r["player_name"], int(r["draft_year"])): i for i, (_, r) in enumerate(df.iterrows())}
        positions = df["position"].values
        updated = 0
        for p in prospects:
            key = (p["player_name"], int(p["draft_year"]))
            if key not in lookup:
                continue
            idx = lookup[key]
            r = df.iloc[idx]
            p["cluster"] = int(r["cluster"])
            p["pca_1"] = round(float(r["pca_1"]), 3)
            p["pca_2"] = round(float(r["pca_2"]), 3)

            dists = np.linalg.norm(X_all_s - X_all_s[idx], axis=1)
            dists[idx] = np.inf
            dists = np.where(positions == r["position"], dists, np.inf)
            comp_idxs = np.argsort(dists)[:5]
            p["comps"] = [{
                "player_name": df.iloc[ci]["player_name"],
                "draft_year":  int(df.iloc[ci]["draft_year"]),
                "college":     df.iloc[ci]["college"],
                "position":    df.iloc[ci]["position"],
                "actual_vorp": round(float(df.iloc[ci]["nba_4yr_vorp"]), 1),
            } for ci in comp_idxs]
            updated += 1
        prospects_path.write_text(json.dumps(prospects))
        print(f"Patched prospects.json ({updated}/{len(prospects)} players re-clustered)")

if __name__ == "__main__":
    main()
