"""
SHAP explainability — deep dive into xG model decisions.
Run after training. Produces:
  - shap_importance.png     (bar chart of mean |SHAP|)
  - shap_beeswarm.png       (beeswarm distribution)
  - shap_dependence_*.png   (distance & angle dependence)
  - shap_waterfall_*.png    (per-shot explanations for interesting shots)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

FEATURES = [
    "distance", "angle", "is_header", "is_volley",
    "is_big_chance", "is_free_kick", "defenders_in_cone",
    "gk_set", "centrality",
]

OUT_DIR   = "models/shap_plots"
DATA_PATH = "data/shots.csv"
MODEL_PATH = "models/xg_model.json"


def load(n_samples: int = 3000) -> tuple[pd.DataFrame, xgb.XGBClassifier]:
    df = pd.read_csv(DATA_PATH)
    df = df[df["is_penalty"] == 0].dropna(subset=FEATURES + ["goal"])

    # Stratified sample to keep goal rate representative
    goals   = df[df["goal"] == 1]
    no_goals = df[df["goal"] == 0].sample(
        min(len(df[df["goal"] == 0]), n_samples - len(goals)),
        random_state=42,
    )
    sample = pd.concat([goals, no_goals]).sample(frac=1, random_state=42)
    X = sample[FEATURES]

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return X, model


def compute_shap(model, X: pd.DataFrame):
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    print(f"SHAP values shape: {sv.shape}")
    return explainer, sv


def plot_summary_bar(sv, X, out_dir):
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X, plot_type="bar", show=False, color="#C9A84C")
    plt.title("Mean absolute SHAP — xG model features", fontsize=12, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_beeswarm(sv, X, out_dir):
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X, show=False, alpha=0.4)
    plt.title("SHAP beeswarm — xG model", fontsize=12, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_beeswarm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_dependence(sv, X, out_dir):
    for feat in ["distance", "angle", "centrality"]:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            feat, sv, X,
            interaction_index="auto",
            show=False,
            dot_size=8,
            alpha=0.5,
        )
        plt.title(f"SHAP dependence — {feat}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(out_dir, f"shap_dep_{feat}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def plot_waterfall_examples(explainer, X: pd.DataFrame, model, out_dir):
    """
    Show waterfall plots for:
    - highest xG shot (best chance)
    - lowest xG shot (most difficult chance)
    - a header from distance
    """
    probs = model.predict_proba(X)[:, 1]
    sv_obj = explainer(X)

    cases = {
        "highest_xg": int(np.argmax(probs)),
        "lowest_xg":  int(np.argmin(probs)),
        "median_xg":  int(np.argsort(probs)[len(probs) // 2]),
    }

    for label, idx in cases.items():
        plt.figure(figsize=(8, 4))
        shap.waterfall_plot(sv_obj[idx], max_display=9, show=False)
        plt.title(f"SHAP waterfall — {label.replace('_', ' ')} (xG={probs[idx]:.3f})", fontsize=11)
        plt.tight_layout()
        path = os.path.join(out_dir, f"shap_waterfall_{label}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def export_feature_importance(sv, X: pd.DataFrame) -> dict:
    mean_abs = np.abs(sv).mean(axis=0)
    importance = {
        feat: round(float(val), 4)
        for feat, val in zip(FEATURES, mean_abs)
    }
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    with open("models/shap_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    print("\nSHAP feature importances:")
    for feat, val in importance.items():
        bar = "#" * int(val * 30)
        print(f"  {feat:<22} {val:.4f}  {bar}")

    return importance


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Run extract.py first — {DATA_PATH} not found.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Run train.py first — {MODEL_PATH} not found.")

    X, model = load(n_samples=3000)
    explainer, sv = compute_shap(model, X)

    plot_summary_bar(sv, X, OUT_DIR)
    plot_beeswarm(sv, X, OUT_DIR)
    plot_dependence(sv, X, OUT_DIR)
    plot_waterfall_examples(explainer, X, model, OUT_DIR)
    export_feature_importance(sv, X)

    print(f"\nAll SHAP plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
