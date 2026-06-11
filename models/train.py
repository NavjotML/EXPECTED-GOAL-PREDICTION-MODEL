"""
xG model training — XGBoost classifier with SHAP explainability.
Input:  data/shots.csv  (produced by extract.py)
Output: models/xg_model.json   (trained XGBoost model)
        models/feature_names.txt
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


FEATURES = [
    "distance",
    "angle",
    "is_header",
    "is_volley",
    "is_big_chance",
    "is_free_kick",
    "defenders_in_cone",
    "gk_set",
    "centrality",
    "dist_angle"
]

MODEL_PARAMS = {
    "n_estimators":      600,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  10,
    "gamma":             1.0,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "scale_pos_weight":  3,   # class imbalance: ~10% goals
    "eval_metric":       "logloss",

    "random_state":      42,
}


def load_data(path: str = "data/shots.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df[df["is_penalty"] == 0].copy()
    df["dist_angle"] = df["distance"] * df["angle"]
    df = df.dropna(subset=FEATURES + ["goal"])
    X = df[FEATURES]
    y = df["goal"]
    print(f"Loaded {len(df)} shots | Goals: {y.sum()} ({y.mean()*100:.1f}%)")
    return X, y


def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc   = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    ll    = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    brier = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score")

    results = {
        "auc_mean":    round(float(auc.mean()), 4),
        "auc_std":     round(float(auc.std()),  4),
        "logloss_mean": round(float(-ll.mean()), 4),
        "brier_mean":  round(float(-brier.mean()), 4),
    }
    print("\nCross-validation results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def train_full(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(
        X, y,
        eval_set=[(X, y)],
        verbose=False,
    )
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc_roc":    round(roc_auc_score(y, probs), 4),
        "log_loss":   round(log_loss(y, probs), 4),
        "brier":      round(brier_score_loss(y, probs), 4),
        "accuracy":   round((preds == y).mean(), 4),
    }
    print("\nFull-dataset metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nClassification report:")
    print(classification_report(y, preds, target_names=["No Goal", "Goal"]))
    return metrics


def plot_shap(model, X: pd.DataFrame, out_dir: str = "models") -> None:
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    # Summary bar plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_vals, X,
        plot_type="bar",
        show=False,
        color="#C9A84C",
    )
    plt.title("SHAP Feature Importance — xG Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_importance.png"), dpi=150)
    plt.close()

    # Beeswarm
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals, X, show=False)
    plt.title("SHAP Beeswarm — xG Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=150)
    plt.close()

    print(f"SHAP plots saved to {out_dir}/")

    # Return top feature importances for API
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance = dict(zip(FEATURES, [round(float(v), 4) for v in mean_abs]))
    return dict(sorted(importance.items(), key=lambda x: -x[1]))


def plot_calibration(model, X: pd.DataFrame, y: pd.Series, out_dir: str = "models") -> None:
    probs = model.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, "o-", color="#C9A84C", label="xG model")
    plt.plot([0, 1], [0, 1], "--", color="#444", label="Perfect calibration")
    plt.xlabel("Mean predicted xG")
    plt.ylabel("Fraction of goals")
    plt.title("Model calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration.png"), dpi=150)
    plt.close()
    print("Calibration plot saved.")


def save_model(model, cv_results: dict, metrics: dict, shap_importance: dict) -> None:
    os.makedirs("models", exist_ok=True)
    model.save_model("models/xg_model.json")

    with open("models/feature_names.txt", "w") as f:
        f.write("\n".join(FEATURES))

    meta = {
        "features":        FEATURES,
        "cv_results":      cv_results,
        "train_metrics":   metrics,
        "shap_importance": shap_importance,
        "model_params":    {k: v for k, v in MODEL_PARAMS.items() if k != "use_label_encoder"},
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nModel saved: models/xg_model.json")
    print("Metadata:    models/model_meta.json")


def main() -> None:
    X, y = load_data()
    cv_results = cross_validate(X, y)
    model      = train_full(X, y)
    metrics    = evaluate(model, X, y)
    shap_imp   = plot_shap(model, X)
    plot_calibration(model, X, y)
    save_model(model, cv_results, metrics, shap_imp)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
