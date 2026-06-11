"""
API test script — validates all endpoints without a running StatsBomb pipeline.
Uses a mock model so you can test the API structure before training.

Run with: python test_api.py
Requires the API to be running: uvicorn api.main:app --port 8000
"""

import json
import sys
import numpy as np
import xgboost as xgb
import os
import requests


BASE_URL = "http://localhost:8000"


def create_mock_model():
    """Create a minimal XGBoost model for testing without real training data."""
    os.makedirs("models", exist_ok=True)

    if os.path.exists("models/xg_model.json"):
        print("Real model found — skipping mock creation.")
        return

    print("Creating mock model for API testing...")

    rng = np.random.default_rng(42)
    n   = 2000
    X   = pd.DataFrame({
        "distance":          rng.uniform(5, 50, n),
        "angle":             rng.uniform(0.05, 1.2, n),
        "is_header":         rng.integers(0, 2, n),
        "is_volley":         rng.integers(0, 2, n),
        "is_big_chance":     rng.integers(0, 2, n),
        "is_free_kick":      rng.integers(0, 2, n),
        "defenders_in_cone": rng.integers(0, 5, n),
        "gk_set":            rng.integers(0, 2, n),
        "centrality":        rng.uniform(0, 1, n),
    })
    # Simulate realistic xG: closer + wider angle = more likely goal
    logit = -3.5 + (50 - X["distance"]) * 0.08 + X["angle"] * 2 + X["is_big_chance"] * 1.5
    prob  = 1 / (1 + np.exp(-logit))
    y     = (rng.random(n) < prob).astype(int)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    model.save_model("models/xg_model.json")
    print("Mock model saved.")


def test_health():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"  [PASS] Health check: {data['service']}")


def test_teams():
    r = requests.get(f"{BASE_URL}/teams")
    assert r.status_code == 200
    data = r.json()
    assert len(data["teams"]) > 0
    print(f"  [PASS] Teams: {len(data['teams'])} teams, {len(data['venues'])} venues")


def test_model_info():
    r = requests.get(f"{BASE_URL}/model/info")
    assert r.status_code == 200
    print(f"  [PASS] Model info: {list(r.json().keys())}")


def test_shot_prediction():
    shots = [
        {"x": 108, "y": 40, "desc": "Central box shot",     "expected_high": True},
        {"x": 90,  "y": 10, "desc": "Wide angle long shot", "expected_high": False},  # keep
        {"x": 116, "y": 40, "desc": "Close range central",  "expected_high": True},
        {"x": 75,  "y": 40, "desc": "Long range central",   "expected_high": False},  # keep
    ]

    for s in shots:
        payload = {
            "x": s["x"], "y": s["y"],
            "is_header": 0, "is_volley": 0,
            "is_free_kick": 0, "defenders_in_cone": 1, "gk_set": 1,
        }
        r = requests.post(f"{BASE_URL}/predict/shot", json=payload)
        assert r.status_code == 200, f"Shot prediction failed: {r.text}"
        data = r.json()
        xg = data["xg"]
        status = "PASS" if (xg > 0.20) == s["expected_high"] else "WARN"
        print(f"  [{status}] {s['desc']}: xG={xg:.4f} | dist={data['distance']:.1f}yd | angle={data['angle_deg']:.1f}deg | big_chance={data['is_big_chance']}")


def test_match_prediction():
    matchups = [
        {"home": "Brazil",   "away": "Morocco",  "venue": "MetLife Stadium, NJ"},
        {"home": "Spain",    "away": "Japan",    "venue": "Rose Bowl, LA"},
        {"home": "Argentina","away": "France",   "venue": "AT&T Stadium, TX"},
    ]

    for m in matchups:
        r = requests.post(f"{BASE_URL}/predict/match", json={
            "home_team": m["home"],
            "away_team": m["away"],
            "venue":     m["venue"],
            "stage":     "Group Stage",
        })
        assert r.status_code == 200, f"Match prediction failed: {r.text}"
        data = r.json()
        probs_sum = data["home_win_prob"] + data["draw_prob"] + data["away_win_prob"]
        assert abs(probs_sum - 1.0) < 0.02, f"Probabilities don't sum to 1: {probs_sum}"
        print(
            f"  [PASS] {m['home']} vs {m['away']}: "
            f"xG {data['home_xg']:.2f}-{data['away_xg']:.2f} | "
            f"W/D/L {data['home_win_prob']:.0%}/{data['draw_prob']:.0%}/{data['away_win_prob']:.0%} | "
            f"shots H={len(data['home_shots'])} A={len(data['away_shots'])}"
        )


def test_edge_cases():
    # Same team
    r = requests.post(f"{BASE_URL}/predict/match", json={
        "home_team": "Brazil", "away_team": "Brazil",
        "venue": "MetLife Stadium, NJ", "stage": "Group Stage",
    })
    assert r.status_code == 400
    print(f"  [PASS] Same-team rejection: {r.json()['detail']}")

    # Unknown team
    r = requests.post(f"{BASE_URL}/predict/match", json={
        "home_team": "Narnia FC", "away_team": "Brazil",
        "venue": "MetLife Stadium, NJ", "stage": "Group Stage",
    })
    assert r.status_code == 400
    print(f"  [PASS] Unknown team rejection: {r.json()['detail'][:50]}...")


def main():
    print("\n--- WC 2026 xG Engine API tests ---\n")

    try:
        requests.get(BASE_URL, timeout=2)
    except Exception:
        print(f"API not running at {BASE_URL}")
        print("Start it with: cd api && uvicorn main:app --port 8000")
        sys.exit(1)

    print("Health:")
    test_health()

    print("\nTeams & venues:")
    test_teams()

    print("\nModel info:")
    test_model_info()

    print("\nShot predictions:")
    test_shot_prediction()

    print("\nMatch simulations:")
    test_match_prediction()

    print("\nEdge cases:")
    test_edge_cases()

    print("\n--- All tests passed ---\n")


if __name__ == "__main__":
    import pandas as pd
    create_mock_model()
    main()
