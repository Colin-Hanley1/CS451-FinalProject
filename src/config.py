from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "fixeddata.csv"
BIGBOARD_CSV = ROOT / "2026class.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
SITE_DATA_DIR = ROOT / "site" / "data"

SPLIT_YEAR = 2019

IDENTITY_COLS = ["player_name", "college", "draft_year", "pick_number"]
TARGET_COLS = [
    "nba_seasons", "nba_4yr_total_mp", "nba_4yr_bpm", "nba_4yr_vorp", "nba_4yr_ws",
    "career_ws48", "meaningful_career", "bpm_tier", "minutes_tier", "PosVORP",
]

POSITION_REL_STATS = ["ts_pct", "spg", "rpg", "apg", "ppg", "bpg", "ws_40"]

POSITION_ORDER = ["Guard", "Forward", "Center"]
K_BY_POSITION = {"Guard": 3, "Forward": 3, "Center": 2}

PROFILE_COLS = ["ppg", "apg", "rpg", "spg", "bpg", "ts_pct", "ws_40",
                "recruit_rank", "college_seasons"]

XGB_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
}

NN_PARAMS = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
    "learning_rate_init": [0.001, 0.005, 0.01],
    "alpha": [0.0001, 0.001, 0.01],
}

RANDOM_STATE = 42
