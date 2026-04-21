from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.config import (FIGURES_DIR, K_BY_POSITION, MODELS_DIR, NN_PARAMS,
                        POSITION_ORDER, PROFILE_COLS, RANDOM_STATE, SITE_DATA_DIR,
                        SPLIT_YEAR, XGB_PARAMS)
from src.preprocess import (compute_position_stats, engineer_features,
                            feature_columns, load_clean_data,
                            three_pt_shrinkage_params)

sns.set_style("whitegrid")

def prepare_data():
    print("─" * 68)
    print("Stage 1-2: Load, clean, engineer features")
    print("─" * 68)
    df = load_clean_data()
    print(f"  Loaded {len(df)} rows after dropping missing target")

    pos_stats = compute_position_stats(df)
    league_3p, k_3p = three_pt_shrinkage_params(df)
    print(f"  Volume-weighted 3P%: league_mean={league_3p:.3f}, k={k_3p:.3f}")

    df_feat = engineer_features(df, pos_stats, league_3p, k_3p)

    if "three_pt_pct" in df_feat.columns:
        df_feat = df_feat.drop(columns=["three_pt_pct"])

    feature_cols = feature_columns(df_feat)
    print(f"  {len(feature_cols)} feature columns after engineering")
    return df_feat, pos_stats, league_3p, k_3p, feature_cols

def temporal_split(df_feat, feature_cols):
    train_df = df_feat[df_feat["draft_year"] < SPLIT_YEAR].copy()
    test_df = df_feat[df_feat["draft_year"] >= SPLIT_YEAR].copy()
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train_vorp = train_df["nba_4yr_vorp"].values
    y_test_vorp = test_df["nba_4yr_vorp"].values
    y_train_cls = train_df["PosVORP"].values
    y_test_cls = test_df["PosVORP"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"  Train: {len(train_df)} players (2000 to {SPLIT_YEAR - 1})")
    print(f"  Test:  {len(test_df)} players ({SPLIT_YEAR} to 2022)")
    return (train_df, test_df, X_train_s, X_test_s,
            y_train_vorp, y_test_vorp, y_train_cls, y_test_cls, scaler)

def train_xgb_regression(X_train_s, y_train_vorp, X_test_s, y_test_vorp):
    print("\n  XGBoost regression (RandomizedSearchCV, n_iter=50)...")
    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        XGB_PARAMS, n_iter=50, cv=5, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train_s, y_train_vorp)
    best = search.best_estimator_
    y_pred = best.predict(X_test_s)
    r2 = r2_score(y_test_vorp, y_pred)
    mae = mean_absolute_error(y_test_vorp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_vorp, y_pred))
    print(f"    CV R² = {search.best_score_:.4f} | Test R² = {r2:.4f} | MAE = {mae:.3f} | RMSE = {rmse:.3f}")
    return best, y_pred, {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}

def train_xgb_classification(X_train_s, y_train_cls, X_test_s, y_test_cls):
    print("\n  XGBoost classification (RandomizedSearchCV, n_iter=50)...")
    search = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=RANDOM_STATE, verbosity=0),
        XGB_PARAMS, n_iter=50, cv=5, scoring="accuracy",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train_s, y_train_cls)
    best = search.best_estimator_
    y_pred = best.predict(X_test_s)
    y_proba = best.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test_cls, y_pred)
    f1 = f1_score(y_test_cls, y_pred)
    auc = roc_auc_score(y_test_cls, y_proba)
    print(f"    Test Accuracy = {acc:.4f} | F1 = {f1:.4f} | AUC-ROC = {auc:.4f}")
    return best, y_pred, y_proba, {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc_roc": round(auc, 4)}

def train_nn_regression(X_train_s, y_train_vorp, X_test_s, y_test_vorp):
    print("\n  Neural Network regression (RandomizedSearchCV, n_iter=20)...")
    search = RandomizedSearchCV(
        MLPRegressor(max_iter=1000, random_state=RANDOM_STATE, early_stopping=True,
                     validation_fraction=0.15),
        NN_PARAMS, n_iter=20, cv=5, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train_s, y_train_vorp)
    best = search.best_estimator_
    y_pred = best.predict(X_test_s)
    r2 = r2_score(y_test_vorp, y_pred)
    mae = mean_absolute_error(y_test_vorp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_vorp, y_pred))
    print(f"    Test R² = {r2:.4f} | MAE = {mae:.3f} | RMSE = {rmse:.3f}")
    return best, y_pred, {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}

def train_nn_classification(X_train_s, y_train_cls, X_test_s, y_test_cls):
    print("\n  Neural Network classification (RandomizedSearchCV, n_iter=20)...")
    search = RandomizedSearchCV(
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE, early_stopping=True,
                      validation_fraction=0.15),
        NN_PARAMS, n_iter=20, cv=5, scoring="accuracy",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train_s, y_train_cls)
    best = search.best_estimator_
    y_pred = best.predict(X_test_s)
    y_proba = best.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test_cls, y_pred)
    f1 = f1_score(y_test_cls, y_pred)
    auc = roc_auc_score(y_test_cls, y_proba)
    print(f"    Test Accuracy = {acc:.4f} | F1 = {f1:.4f} | AUC-ROC = {auc:.4f}")
    return best, {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc_roc": round(auc, 4)}

def train_ridge(X_train_s, y_train_vorp, X_test_s, y_test_vorp):
    print("\n  Ridge baseline...")
    ridge = Ridge(alpha=10).fit(X_train_s, y_train_vorp)
    y_pred = ridge.predict(X_test_s)
    r2 = r2_score(y_test_vorp, y_pred)
    mae = mean_absolute_error(y_test_vorp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_vorp, y_pred))
    print(f"    Test R² = {r2:.4f} | MAE = {mae:.3f} | RMSE = {rmse:.3f}")
    return ridge, y_pred, {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}

def cluster_and_project(df_feat, feature_cols, scaler):
    print("─" * 68)
    print("Stage 4: Per-position K-means + global PCA")
    print("─" * 68)

    X_all_s = scaler.transform(df_feat[feature_cols].values)

    km_models = {}
    global_cluster = np.full(len(df_feat), -1, dtype=int)
    offset = 0
    silhouettes = {}
    for pos in POSITION_ORDER:
        k = K_BY_POSITION[pos]
        mask = (df_feat["position"].values == pos)
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_all_s[mask])
        global_cluster[mask] = km.labels_ + offset
        km_models[pos] = {"model": km, "offset": offset, "k": k}
        sil = silhouette_score(X_all_s[mask], km.labels_)
        silhouettes[pos] = sil
        print(f"  {pos:8s}: k={k}, n={mask.sum()}, silhouette={sil:.3f}")
        offset += k

    df_feat = df_feat.copy()
    df_feat["cluster"] = global_cluster

    pca = PCA(n_components=2).fit(X_all_s)
    pcs = pca.transform(X_all_s)
    df_feat["pca_1"] = pcs[:, 0]
    df_feat["pca_2"] = pcs[:, 1]
    print(f"  PCA variance explained: PC1 = {pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2 = {pca.explained_variance_ratio_[1]:.1%}")

    return df_feat, km_models, pca, X_all_s

def save_model_comparison_figure(reg_results, cls_results, y_test_cls, y_pred_cls_xgb):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = list(reg_results.keys())
    r2s = [reg_results[m]["r2"] for m in models]
    maes = [reg_results[m]["mae"] / 10 for m in models]
    rmses = [reg_results[m]["rmse"] / 10 for m in models]
    x = np.arange(len(models)); w = 0.3
    axes[0].bar(x - w, r2s, w, label="R²", color="#2c5f8a", alpha=0.85)
    axes[0].bar(x, maes, w, label="MAE (/10)", color="#5a8a3c", alpha=0.85)
    axes[0].bar(x + w, rmses, w, label="RMSE (/10)", color="#c0392b", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].legend(); axes[0].set_title("Regression: VORP Prediction")

    cm = confusion_matrix(y_test_cls, y_pred_cls_xgb)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Neg VORP", "Pos VORP"], yticklabels=["Neg VORP", "Pos VORP"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title("XGBoost Classification: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_model_comparison.png", bbox_inches="tight")
    plt.close()

def save_shap_figure(xgb_reg, X_test_s, feature_cols):
    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_test_s)
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_s, feature_names=feature_cols, show=False, max_display=15)
    plt.title("SHAP Feature Importance (XGBoost Regression)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_shap_importance.png", bbox_inches="tight")
    plt.close()

def save_kmeans_figure(df_feat, km_models, pca):
    cluster_colors = ["#a63818", "#1f3a5c", "#516b3e", "#b9862c",
                      "#2d6e73", "#6a2e52", "#4a4a5a", "#8b2014"]
    markers = {"Guard": "o", "Forward": "s", "Center": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    total_k = sum(K_BY_POSITION.values())

    for c in range(total_k):
        mask = df_feat["cluster"] == c
        if mask.sum() == 0:
            continue
        pos = df_feat.loc[mask, "position"].iloc[0]
        axes[0].scatter(df_feat.loc[mask, "pca_1"], df_feat.loc[mask, "pca_2"],
                        s=18, alpha=0.55, color=cluster_colors[c % len(cluster_colors)],
                        marker=markers[pos], label=f"Cluster {c} ({pos[0]})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    axes[0].set_title("Player Archetypes (per-position K-Means + global PCA)")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")

    cluster_vorp = df_feat.groupby("cluster")["nba_4yr_vorp"].agg(["mean", "count"])
    cluster_vorp.plot.bar(y="mean", ax=axes[1], color=cluster_colors[:total_k],
                          alpha=0.75, legend=False)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("Mean 4-Year VORP")
    axes[1].set_title("NBA Outcomes by Archetype")
    for i, row in cluster_vorp.iterrows():
        axes[1].text(i, row["mean"] + 0.1, f"n={row['count']:.0f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_kmeans_clusters.png", bbox_inches="tight")
    plt.close()

def export_prospects_json(df_feat, feature_cols, scaler, xgb_reg, xgb_clf, X_all_s):
    print("\n  Computing predictions + SHAP for all prospects...")
    all_vorp_pred = xgb_reg.predict(X_all_s)
    all_cls_proba = xgb_clf.predict_proba(X_all_s)[:, 1]
    explainer = shap.TreeExplainer(xgb_reg)
    all_shap = explainer.shap_values(X_all_s)
    positions = df_feat["position"].values

    prospects = []
    for i, (_, row) in enumerate(df_feat.iterrows()):
        shap_pairs = sorted(
            [(feature_cols[j], float(all_shap[i][j])) for j in range(len(feature_cols))],
            key=lambda x: abs(x[1]), reverse=True,
        )[:5]

        dists = np.linalg.norm(X_all_s - X_all_s[i], axis=1)
        dists[i] = np.inf
        dists = np.where(positions == row["position"], dists, np.inf)
        comp_idx = np.argsort(dists)[:5]
        comps = [{
            "player_name": df_feat.iloc[ci]["player_name"],
            "draft_year": int(df_feat.iloc[ci]["draft_year"]),
            "college": df_feat.iloc[ci]["college"],
            "position": df_feat.iloc[ci]["position"],
            "actual_vorp": round(float(df_feat.iloc[ci]["nba_4yr_vorp"]), 1),
        } for ci in comp_idx]

        prospects.append({
            "player_name": row["player_name"],
            "draft_year": int(row["draft_year"]),
            "college": row["college"],
            "position": row["position"],
            "ppg": round(float(row["ppg"]), 1),
            "apg": round(float(row["apg"]), 1),
            "rpg": round(float(row["rpg"]), 1),
            "spg": round(float(row["spg"]), 2),
            "bpg": round(float(row["bpg"]), 2),
            "ts_pct": round(float(row["ts_pct"]), 3),
            "ws_40": round(float(row["ws_40"]), 3),
            "recruit_rank": int(row["recruit_rank"]),
            "college_seasons": int(row["college_seasons"]),
            "actual_vorp": round(float(row["nba_4yr_vorp"]), 2),
            "predicted_vorp": round(float(all_vorp_pred[i]), 2),
            "posvorp_prob": round(float(all_cls_proba[i]), 3),
            "cluster": int(row["cluster"]),
            "pca_1": round(float(row["pca_1"]), 3),
            "pca_2": round(float(row["pca_2"]), 3),
            "shap_features": [{"feature": f, "shap": round(s, 3)} for f, s in shap_pairs],
            "comps": comps,
            "is_test": bool(row["draft_year"] >= SPLIT_YEAR),
        })

    (SITE_DATA_DIR / "prospects.json").write_text(json.dumps(prospects))
    print(f"  Wrote prospects.json ({len(prospects)} players)")

def export_clusters_json(df_feat, km_models, pca):
    cluster_profiles = {}
    for pos in POSITION_ORDER:
        off = km_models[pos]["offset"]
        for local in range(km_models[pos]["k"]):
            gid = off + local
            mask = df_feat["cluster"] == gid
            if mask.sum() == 0:
                continue
            profile = df_feat.loc[mask, PROFILE_COLS].mean()
            cluster_profiles[str(gid)] = {
                **{col: round(float(profile[col]), 2) for col in PROFILE_COLS},
                "position": pos,
                "local_id": local,
                "count": int(mask.sum()),
                "mean_vorp": round(float(df_feat.loc[mask, "nba_4yr_vorp"].mean()), 2),
            }

    data = {
        "n_clusters": sum(K_BY_POSITION.values()),
        "clustered_by_position": True,
        "k_per_position": K_BY_POSITION,
        "pca_variance_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_[:2]],
        "cluster_profiles": cluster_profiles,
    }
    (SITE_DATA_DIR / "clusters.json").write_text(json.dumps(data, indent=2))
    print(f"  Wrote clusters.json ({data['n_clusters']} archetypes)")

def export_position_stats_json(pos_stats):

    frontend_stats = {k: v for k, v in pos_stats.items() if k not in ("height_inches", "weight_lbs")}
    (SITE_DATA_DIR / "position_stats.json").write_text(json.dumps(frontend_stats, indent=2))
    print("  Wrote position_stats.json")

def export_model_meta_json(df_feat, feature_cols, train_df, test_df, reg_results,
                           cls_results, xgb_reg, X_all_s, scaler):

    ridge_full = Ridge(alpha=10).fit(X_all_s, df_feat["nba_4yr_vorp"].values)

    explainer = shap.TreeExplainer(xgb_reg)
    shap_values = explainer.shap_values(X_all_s)
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)

    meta = {
        "n_training": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "split_year": SPLIT_YEAR,
        "regression_results": reg_results,
        "classification_results": cls_results,
        "feature_importances": {feat: round(float(v), 4) for feat, v in shap_importance.head(15).items()},
        "ridge_coefficients": {feat: round(float(c), 4) for feat, c in zip(feature_cols, ridge_full.coef_)},
        "ridge_intercept": round(float(ridge_full.intercept_), 4),
        "scaler_mean": {feat: round(float(m), 6) for feat, m in zip(feature_cols, scaler.mean_)},
        "scaler_scale": {feat: round(float(s), 6) for feat, s in zip(feature_cols, scaler.scale_)},
    }
    (SITE_DATA_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("  Wrote model_meta.json")

def save_artifacts(xgb_reg, xgb_clf, nn_reg, nn_clf, ridge, scaler, km_models):
    joblib.dump(xgb_reg, MODELS_DIR / "xgboost_reg.pkl")
    joblib.dump(xgb_clf, MODELS_DIR / "xgboost_clf.pkl")
    joblib.dump(nn_reg, MODELS_DIR / "nn_reg.pkl")
    joblib.dump(nn_clf, MODELS_DIR / "nn_clf.pkl")
    joblib.dump(ridge, MODELS_DIR / "ridge.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(km_models, MODELS_DIR / "kmeans.pkl")
    print("  Saved 7 artifacts to models/")

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_feat, pos_stats, _, _, feature_cols = prepare_data()

    print("─" * 68)
    print("Stage 3: Train models (temporal split at draft year "
          f"{SPLIT_YEAR})")
    print("─" * 68)
    (train_df, test_df, X_train_s, X_test_s,
     y_train_vorp, y_test_vorp, y_train_cls, y_test_cls, scaler) = temporal_split(
        df_feat, feature_cols)

    xgb_reg, _, xgb_reg_metrics = train_xgb_regression(X_train_s, y_train_vorp, X_test_s, y_test_vorp)
    xgb_clf, y_pred_cls_xgb, _, xgb_cls_metrics = train_xgb_classification(X_train_s, y_train_cls, X_test_s, y_test_cls)
    nn_reg, _, nn_reg_metrics = train_nn_regression(X_train_s, y_train_vorp, X_test_s, y_test_vorp)
    nn_clf, nn_cls_metrics = train_nn_classification(X_train_s, y_train_cls, X_test_s, y_test_cls)
    ridge, _, ridge_metrics = train_ridge(X_train_s, y_train_vorp, X_test_s, y_test_vorp)

    reg_results = {
        "Ridge (baseline)": ridge_metrics,
        "XGBoost": xgb_reg_metrics,
        "Neural Network": nn_reg_metrics,
    }
    cls_results = {
        "XGBoost": xgb_cls_metrics,
        "Neural Network": nn_cls_metrics,
    }

    df_feat, km_models, pca, X_all_s = cluster_and_project(df_feat, feature_cols, scaler)

    print("\n  Saving modeling figures...")
    save_model_comparison_figure(reg_results, cls_results, y_test_cls, y_pred_cls_xgb)
    save_shap_figure(xgb_reg, X_test_s, feature_cols)
    save_kmeans_figure(df_feat, km_models, pca)

    print("─" * 68)
    print("Stage 5: Export artifacts")
    print("─" * 68)
    save_artifacts(xgb_reg, xgb_clf, nn_reg, nn_clf, ridge, scaler, km_models)
    export_prospects_json(df_feat, feature_cols, scaler, xgb_reg, xgb_clf, X_all_s)
    export_clusters_json(df_feat, km_models, pca)
    export_position_stats_json(pos_stats)
    export_model_meta_json(df_feat, feature_cols, train_df, test_df, reg_results,
                           cls_results, xgb_reg, X_all_s, scaler)

    print("\n" + "═" * 68)
    print("Pipeline complete. Now run:")
    print("  python src/generate_bigboard.py   # score the 2026 class")
    print("  python src/eda.py                 # regenerate EDA figures")
    print("  cd site && python -m http.server 8080   # serve the dashboard")
    print("═" * 68)

if __name__ == "__main__":
    main()
