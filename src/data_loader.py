"""
data_loader.py — Data Loading & Merging Pipeline for F1 Race Outcome Predictor
Loads 14 CSV files from the Ergast-style dataset and merges them into a unified DataFrame.
"""

import pandas as pd
import numpy as np
import os
from src.utils import DATA_DIR, MODERN_ERA_START


def load_raw_tables() -> dict:
    """Load all raw CSV files into a dictionary of DataFrames."""
    tables = {}
    csv_files = [
        "circuits", "constructor_results", "constructor_standings",
        "constructors", "driver_standings", "drivers", "lap_times",
        "pit_stops", "qualifying", "races", "results", "seasons",
        "sprint_results", "status"
    ]
    for name in csv_files:
        filepath = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, na_values=["\\N", ""])
            tables[name] = df
    return tables


def merge_race_data(tables: dict) -> pd.DataFrame:
    """
    Merge all relevant tables into a single unified race results DataFrame.
    
    Joins:
      results ← races (raceId) ← circuits (circuitId)
      results ← drivers (driverId)
      results ← constructors (constructorId)
      results ← qualifying (raceId + driverId)
      results ← status (statusId)
    """
    results = tables["results"].copy()
    races = tables["races"].copy()
    circuits = tables["circuits"].copy()
    drivers = tables["drivers"].copy()
    constructors = tables["constructors"].copy()
    qualifying = tables["qualifying"].copy()
    status = tables["status"].copy()

    # ── Merge races + circuits ───────────────────────────────────────────
    races_full = races.merge(
        circuits[["circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt"]],
        on="circuitId",
        how="left",
        suffixes=("", "_circuit")
    )
    races_full.rename(columns={
        "name": "race_name",
        "name_circuit": "circuit_name",
    }, inplace=True)

    # ── Merge results + races ────────────────────────────────────────────
    df = results.merge(
        races_full[["raceId", "year", "round", "circuitId", "race_name", "date",
                     "circuitRef", "circuit_name", "location", "country", "lat", "lng", "alt"]],
        on="raceId",
        how="left"
    )

    # ── Merge drivers ────────────────────────────────────────────────────
    df = df.merge(
        drivers[["driverId", "driverRef", "code", "forename", "surname", "dob", "nationality"]],
        on="driverId",
        how="left",
        suffixes=("", "_driver")
    )
    df["driver_name"] = df["forename"] + " " + df["surname"]
    df.rename(columns={"nationality": "driver_nationality"}, inplace=True)

    # ── Merge constructors ───────────────────────────────────────────────
    df = df.merge(
        constructors[["constructorId", "constructorRef", "name", "nationality"]],
        on="constructorId",
        how="left",
        suffixes=("", "_constructor")
    )
    df.rename(columns={
        "name": "constructor_name",
        "nationality": "constructor_nationality"
    }, inplace=True)

    # ── Merge qualifying ─────────────────────────────────────────────────
    if "qualifying" in tables and not qualifying.empty:
        quali_cols = ["raceId", "driverId", "position", "q1", "q2", "q3"]
        quali_sub = qualifying[quali_cols].copy()
        quali_sub.rename(columns={"position": "quali_position"}, inplace=True)
        df = df.merge(quali_sub, on=["raceId", "driverId"], how="left")

    # ── Merge status ─────────────────────────────────────────────────────
    df = df.merge(status, on="statusId", how="left")

    # ── Filter to modern era ─────────────────────────────────────────────
    df = df[df["year"] >= MODERN_ERA_START].copy()

    # ── Clean up data types ──────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["laps"] = pd.to_numeric(df["laps"], errors="coerce")
    df["milliseconds"] = pd.to_numeric(df["milliseconds"], errors="coerce")
    df["quali_position"] = pd.to_numeric(df.get("quali_position"), errors="coerce")

    # ── Sort by date and position ────────────────────────────────────────
    df.sort_values(["date", "positionOrder"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_driver_standings(tables: dict) -> pd.DataFrame:
    """Get driver championship standings data."""
    ds = tables["driver_standings"].copy()
    races = tables["races"][["raceId", "year", "round"]].copy()
    ds = ds.merge(races, on="raceId", how="left")
    return ds


def get_constructor_standings(tables: dict) -> pd.DataFrame:
    """Get constructor championship standings data."""
    cs = tables["constructor_standings"].copy()
    races = tables["races"][["raceId", "year", "round"]].copy()
    cs = cs.merge(races, on="raceId", how="left")
    return cs


def load_data() -> pd.DataFrame:
    """Main entry point: load and merge all data."""
    tables = load_raw_tables()
    df = merge_race_data(tables)
    return df


def load_all() -> tuple:
    """Load merged data + raw tables for additional queries."""
    tables = load_raw_tables()
    df = merge_race_data(tables)
    return df, tables


if __name__ == "__main__":
    df = load_data()
    print(f"✅ Loaded {len(df)} race results")
    print(f"📅 Date range: {df['year'].min()} – {df['year'].max()}")
    print(f"🏎️  Drivers: {df['driver_name'].nunique()}")
    print(f"🏗️  Constructors: {df['constructor_name'].nunique()}")
    print(f"🏁 Races: {df['raceId'].nunique()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df[['year', 'race_name', 'driver_name', 'constructor_name', 'grid', 'position', 'points']].head(10)}")
