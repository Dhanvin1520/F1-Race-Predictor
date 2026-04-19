"""
feature_engineering.py — Feature Engineering for F1 Race Outcome Predictor
Creates predictive features from raw race data for ML model training.
"""

import pandas as pd
import numpy as np
from src.data_loader import load_data


def parse_quali_time_to_ms(time_str) -> float:
    """Convert qualifying time string (e.g., '1:26.572') to milliseconds."""
    if pd.isna(time_str) or time_str == "" or time_str is None:
        return np.nan
    try:
        time_str = str(time_str).strip()
        if ":" in time_str:
            parts = time_str.split(":")
            minutes = float(parts[0])
            seconds = float(parts[1])
            return (minutes * 60 + seconds) * 1000
        else:
            return float(time_str) * 1000
    except (ValueError, IndexError):
        return np.nan


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all predictive features from the merged race data.
    
    Features are computed using only data available BEFORE each race
    (no data leakage from future races).
    """
    df = df.copy()
    df.sort_values(["date", "positionOrder"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Target Variables ─────────────────────────────────────────────────
    df["is_winner"] = (df["positionOrder"] == 1).astype(int)
    df["is_podium"] = (df["positionOrder"] <= 3).astype(int)
    df["finished"] = (df["status"] == "Finished").astype(int)

    # ── Grid / Qualifying Features ───────────────────────────────────────
    # Handle grid position 0 (pit lane start) → set to 20
    df["grid_clean"] = df["grid"].replace(0, 20)

    # Qualifying best time in ms
    for q_col in ["q1", "q2", "q3"]:
        if q_col in df.columns:
            df[f"{q_col}_ms"] = df[q_col].apply(parse_quali_time_to_ms)

    # Best qualifying time
    q_ms_cols = [c for c in ["q3_ms", "q2_ms", "q1_ms"] if c in df.columns]
    if q_ms_cols:
        df["best_quali_ms"] = df[q_ms_cols].min(axis=1)

    # Gap to pole position (in ms)
    if "best_quali_ms" in df.columns:
        pole_times = df.groupby("raceId")["best_quali_ms"].transform("min")
        df["quali_gap_to_pole"] = df["best_quali_ms"] - pole_times
        df["quali_gap_to_pole"] = df["quali_gap_to_pole"].fillna(
            df.groupby("raceId")["quali_gap_to_pole"].transform("max") * 1.2
        )

    # ── Season Context Features ──────────────────────────────────────────
    df["round_number"] = df["round"]
    max_round_per_year = df.groupby("year")["round"].transform("max")
    df["season_progress"] = df["round"] / max_round_per_year

    # ── Rolling Performance Features (per driver) ────────────────────────
    # Sort by date for proper rolling calculations
    df.sort_values("date", inplace=True)

    # Driver rolling average finish (last 5 races)
    df["driver_avg_finish_last5"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Driver rolling points (last 5 races)
    df["driver_points_last5"] = (
        df.groupby("driverId")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # Driver points momentum (last 3 vs previous 3)
    df["driver_recent3"] = (
        df.groupby("driverId")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["driver_prev3"] = (
        df.groupby("driverId")["points"]
        .transform(lambda x: x.shift(4).rolling(3, min_periods=1).mean())
    )
    df["driver_points_momentum"] = df["driver_recent3"] - df["driver_prev3"].fillna(0)

    # Driver career stats (cumulative up to the race before)
    df["driver_career_races"] = (
        df.groupby("driverId").cumcount()
    )
    df["driver_career_wins"] = (
        df.groupby("driverId")["is_winner"]
        .transform(lambda x: x.shift(1).cumsum())
    )
    df["driver_win_rate"] = (
        df["driver_career_wins"] / df["driver_career_races"].clip(lower=1)
    )

    # Driver DNF rate
    df["driver_career_dnf"] = (
        df.groupby("driverId")["finished"]
        .transform(lambda x: (1 - x).shift(1).cumsum())
    )
    df["driver_dnf_rate"] = (
        df["driver_career_dnf"] / df["driver_career_races"].clip(lower=1)
    )

    # ── Rolling Performance Features (per constructor) ───────────────────
    df["constructor_avg_finish_last5"] = (
        df.groupby("constructorId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    df["constructor_points_last5"] = (
        df.groupby("constructorId")["points"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    )

    # Constructor DNF rate
    df["constructor_total_races"] = df.groupby("constructorId").cumcount()
    df["constructor_career_dnf"] = (
        df.groupby("constructorId")["finished"]
        .transform(lambda x: (1 - x).shift(1).cumsum())
    )
    df["constructor_dnf_rate"] = (
        df["constructor_career_dnf"] / df["constructor_total_races"].clip(lower=1)
    )

    # Constructor win rate
    df["constructor_career_wins"] = (
        df.groupby("constructorId")["is_winner"]
        .transform(lambda x: x.shift(1).cumsum())
    )
    df["constructor_win_rate"] = (
        df["constructor_career_wins"] / df["constructor_total_races"].clip(lower=1)
    )

    # Constructor points momentum
    df["constructor_recent3"] = (
        df.groupby("constructorId")["points"]
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    )
    df["constructor_points_momentum"] = df["constructor_recent3"].fillna(0)

    # ── Circuit-Specific Features (driver at this track) ─────────────────
    df["circuit_key"] = df["driverId"].astype(str) + "_" + df["circuitId"].astype(str)
    df["driver_circuit_starts"] = df.groupby("circuit_key").cumcount()
    df["driver_circuit_avg_finish"] = (
        df.groupby("circuit_key")["positionOrder"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["driver_circuit_best"] = (
        df.groupby("circuit_key")["positionOrder"]
        .transform(lambda x: x.shift(1).expanding().min())
    )

    # ── Championship Position Features ───────────────────────────────────
    # Approximate from cumulative season points
    df["season_points_cum"] = (
        df.groupby(["driverId", "year"])["points"]
        .transform(lambda x: x.shift(1).cumsum())
    ).fillna(0)

    # Rank within year
    df["driver_championship_pos"] = (
        df.groupby(["raceId"])["season_points_cum"]
        .rank(ascending=False, method="min")
    )

    # Constructor championship position
    df["constructor_season_pts"] = (
        df.groupby(["constructorId", "year"])["points"]
        .transform(lambda x: x.shift(1).cumsum())
    ).fillna(0)
    df["constructor_championship_pos"] = (
        df.groupby(["raceId"])["constructor_season_pts"]
        .rank(ascending=False, method="min")
    )

    # ── Clean up temporary columns ───────────────────────────────────────
    drop_cols = ["driver_recent3", "driver_prev3", "circuit_key",
                 "constructor_recent3", "q1_ms", "q2_ms", "q3_ms"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # Re-sort by date and position
    df.sort_values(["date", "positionOrder"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ─── Feature list for model training ────────────────────────────────────
FEATURE_COLUMNS = [
    "grid_clean",
    "quali_position",
    "quali_gap_to_pole",
    "round_number",
    "season_progress",
    "driver_avg_finish_last5",
    "driver_points_last5",
    "driver_points_momentum",
    "driver_career_races",
    "driver_career_wins",
    "driver_win_rate",
    "driver_dnf_rate",
    "constructor_avg_finish_last5",
    "constructor_points_last5",
    "constructor_dnf_rate",
    "constructor_win_rate",
    "constructor_points_momentum",
    "driver_circuit_starts",
    "driver_circuit_avg_finish",
    "driver_circuit_best",
    "driver_championship_pos",
    "constructor_championship_pos",
]


def get_training_data(df: pd.DataFrame, target: str = "is_winner"):
    """
    Prepare X (features) and y (target) for model training.
    Drops rows with missing target or excessive missing features.
    """
    feature_df = df[FEATURE_COLUMNS + [target, "year", "raceId"]].copy()

    # Drop rows where target is NaN
    feature_df = feature_df.dropna(subset=[target])

    # Fill remaining NaN in features with sensible defaults
    for col in FEATURE_COLUMNS:
        if col in feature_df.columns:
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val if not pd.isna(median_val) else 0)

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df[target]
    meta = feature_df[["year", "raceId"]]

    return X, y, meta


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Raw data: {len(df)} rows")

    print("\nEngineering features...")
    df_feat = engineer_features(df)
    print(f"Featured data: {len(df_feat)} rows")

    X, y, meta = get_training_data(df_feat, target="is_winner")
    print(f"\nTraining data shape: X={X.shape}, y={y.shape}")
    print(f"Winner rate: {y.mean():.4f}")
    print(f"Features: {list(X.columns)}")
    print(f"\nFeature sample:\n{X.head()}")
