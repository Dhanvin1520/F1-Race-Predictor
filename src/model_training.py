"""
model_training.py — ML Model Training for F1 Race Outcome Predictor
Trains Random Forest, XGBoost, and LightGBM models for race prediction.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_absolute_error, mean_squared_error, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from src.data_loader import load_data
from src.feature_engineering import engineer_features, get_training_data, FEATURE_COLUMNS
from src.utils import MODEL_DIR

warnings.filterwarnings("ignore")


# ─── Time-Based Train/Test Split ────────────────────────────────────────
TRAIN_END_YEAR = 2023     # Train on 2009–2023
VAL_YEAR = 2024            # Validate on 2024
TEST_START_YEAR = 2025     # Test on 2025–2026


def split_data(X, y, meta):
    """Time-based split: train on historical, test on recent seasons."""
    train_mask = meta["year"] <= TRAIN_END_YEAR
    val_mask = meta["year"] == VAL_YEAR
    test_mask = meta["year"] >= TEST_START_YEAR

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─── Classification Models (Winner / Podium Prediction) ─────────────────
def get_classification_models():
    """Return dictionary of classification models to train."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
    }


# ─── Regression Models (Position Prediction) ────────────────────────────
def get_regression_models():
    """Return dictionary of regression models to train."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mae",
            verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    }


def evaluate_classifier(model, X_test, y_test, model_name="Model"):
    """Evaluate a classification model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def evaluate_regressor(model, X_test, y_test, model_name="Model"):
    """Evaluate a regression model and return metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "model": model_name,
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return metrics


def train_all_models(df_featured: pd.DataFrame):
    """
    Train all models for both classification and regression tasks.
    Returns trained models and evaluation metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {"classification": {}, "regression": {}}
    all_models = {}
    scaler = StandardScaler()

    # ── TASK 1: Winner Prediction (Classification) ───────────────────────
    print("\n" + "=" * 60)
    print("🏆 TASK 1: Winner Prediction (Classification)")
    print("=" * 60)

    X, y, meta = get_training_data(df_featured, target="is_winner")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, meta)

    # Scale features
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Winner rate — Train: {y_train.mean():.4f} | Test: {y_test.mean():.4f}")

    clf_models = get_classification_models()
    best_clf_auc = 0
    best_clf_name = None

    for name, model in clf_models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        metrics = evaluate_classifier(model, X_test_scaled, y_test, name)
        results["classification"][name] = metrics

        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:  {metrics['roc_auc']:.4f}")

        if metrics["roc_auc"] > best_clf_auc:
            best_clf_auc = metrics["roc_auc"]
            best_clf_name = name

        # Save model
        model_path = os.path.join(MODEL_DIR, f"clf_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        all_models[f"clf_{name}"] = model

    print(f"\n  🥇 Best Classifier: {best_clf_name} (AUC: {best_clf_auc:.4f})")

    # ── TASK 2: Position Prediction (Regression) ─────────────────────────
    print("\n" + "=" * 60)
    print("📊 TASK 2: Position Prediction (Regression)")
    print("=" * 60)

    X_reg, y_reg, meta_reg = get_training_data(df_featured, target="positionOrder")
    X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r = split_data(X_reg, y_reg, meta_reg)

    scaler_reg = StandardScaler()
    X_train_r_scaled = pd.DataFrame(scaler_reg.fit_transform(X_train_r), columns=X_train_r.columns, index=X_train_r.index)
    X_val_r_scaled = pd.DataFrame(scaler_reg.transform(X_val_r), columns=X_val_r.columns, index=X_val_r.index)
    X_test_r_scaled = pd.DataFrame(scaler_reg.transform(X_test_r), columns=X_test_r.columns, index=X_test_r.index)

    print(f"  Train: {len(X_train_r)} | Val: {len(X_val_r)} | Test: {len(X_test_r)}")

    reg_models = get_regression_models()
    best_reg_mae = 999
    best_reg_name = None

    for name, model in reg_models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_r_scaled, y_train_r)

        metrics = evaluate_regressor(model, X_test_r_scaled, y_test_r, name)
        results["regression"][name] = metrics

        print(f"    MAE:  {metrics['mae']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")

        if metrics["mae"] < best_reg_mae:
            best_reg_mae = metrics["mae"]
            best_reg_name = name

        model_path = os.path.join(MODEL_DIR, f"reg_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        all_models[f"reg_{name}"] = model

    print(f"\n  🥇 Best Regressor: {best_reg_name} (MAE: {best_reg_mae:.4f})")

    # ── Save scalers ─────────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_clf.pkl"))
    joblib.dump(scaler_reg, os.path.join(MODEL_DIR, "scaler_reg.pkl"))

    # ── Save feature importances ─────────────────────────────────────────
    feature_importances = {}
    for model_key, model in all_models.items():
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
            feature_importances[model_key] = fi

    # ── Save all results ─────────────────────────────────────────────────
    results["feature_importances"] = feature_importances
    results["best_classifier"] = best_clf_name
    results["best_regressor"] = best_reg_name
    results["data_info"] = {
        "train_years": f"2009-{TRAIN_END_YEAR}",
        "val_year": str(VAL_YEAR),
        "test_years": f"{TEST_START_YEAR}-2026",
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }

    results_path = os.path.join(MODEL_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_path}")

    return all_models, results


def load_trained_models():
    """Load trained models from disk."""
    models = {}
    model_files = {
        "clf_Random Forest": "clf_random_forest.pkl",
        "clf_XGBoost": "clf_xgboost.pkl",
        "clf_LightGBM": "clf_lightgbm.pkl",
        "reg_Random Forest": "reg_random_forest.pkl",
        "reg_XGBoost": "reg_xgboost.pkl",
        "reg_LightGBM": "reg_lightgbm.pkl",
        "scaler_clf": "scaler_clf.pkl",
        "scaler_reg": "scaler_reg.pkl",
    }
    for key, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models


def load_results():
    """Load training results from disk."""
    results_path = os.path.join(MODEL_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    print("🏎️  F1 Race Outcome Predictor — Model Training")
    print("=" * 60)

    print("\n📥 Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} race results")

    print("\n⚙️  Engineering features...")
    df_feat = engineer_features(df)
    print(f"   Featured data: {len(df_feat)} rows")

    print("\n🧠 Training models...")
    models, results = train_all_models(df_feat)

    print("\n" + "=" * 60)
    print("✅ All models trained and saved!")
    print("=" * 60)
