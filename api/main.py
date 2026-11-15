"""
FastAPI xG Engine — WC 2026 (simplified)
Endpoints:
  GET  /                      health check
  GET  /teams                 list of WC 2026 teams with style profiles
  POST /predict/shot          xG for a single shot
  POST /predict/match         team xG totals + simulated shot locations
  GET  /model/info            model metadata + SHAP importances
"""

from __future__ import annotations
import json
import os
import random
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WC 2026 xG Engine",
    description="Expected Goals prediction using XGBoost trained on StatsBomb open data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xg_model.json")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "model_meta.json")

_model: xgb.XGBClassifier | None = None
_meta:  dict | None = None

FEATURES = [
    "distance", "angle", "is_header", "is_volley",
    "is_big_chance", "is_free_kick", "defenders_in_cone",
    "gk_set", "centrality", "dist_angle"
]


def get_model() -> xgb.XGBClassifier:
    """
    Loads the trained xG model from disk the first time it's needed,
    then keeps it in memory so we don't reload it on every single request.
    Think of it like opening a heavy book once and keeping it on your desk,
    instead of walking to the shelf every time you need to read a line.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run models/train.py first."
            )
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
    return _model


def get_meta() -> dict:
    """
    Loads the model's 'report card' — accuracy scores and which features
    mattered most — from a JSON file, and caches it the same way get_model()
    does. This is what powers the /model/info endpoint.
    """
    global _meta
    if _meta is None and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _meta or {}


# ---------------------------------------------------------------------------
# Team profiles
# ---------------------------------------------------------------------------
# A hand-written lookup table, not something the model learned. For each
# team it just stores a rough playing style label and a plausible xG range
# per match, used later to decide how many shots to simulate for that team.

TEAMS: dict[str, dict] = {
    "Argentina":   {"style": "attack",     "xg_range": [1.6, 2.2]},
    "Brazil":      {"style": "attack",     "xg_range": [1.5, 2.1]},
    "France":      {"style": "balanced",   "xg_range": [1.2, 1.9]},
    "England":     {"style": "balanced",   "xg_range": [1.1, 1.8]},
    "Spain":       {"style": "possession", "xg_range": [1.3, 1.9]},
    "Germany":     {"style": "balanced",   "xg_range": [1.2, 1.8]},
    "Portugal":    {"style": "attack",     "xg_range": [1.4, 2.0]},
    "Netherlands": {"style": "balanced",   "xg_range": [1.1, 1.7]},
    "Morocco":     {"style": "defensive",  "xg_range": [0.7, 1.2]},
    "USA":         {"style": "balanced",   "xg_range": [0.9, 1.6]},
    "Mexico":      {"style": "defensive",  "xg_range": [0.8, 1.3]},
    "Canada":      {"style": "balanced",   "xg_range": [0.9, 1.5]},
    "Japan":       {"style": "pressing",   "xg_range": [0.9, 1.4]},
    "South Korea": {"style": "pressing",   "xg_range": [0.8, 1.4]},
    "Senegal":     {"style": "balanced",   "xg_range": [1.0, 1.6]},
    "Uruguay":     {"style": "defensive",  "xg_range": [0.8, 1.3]},
    "Croatia":     {"style": "defensive",  "xg_range": [0.8, 1.3]},
    "Belgium":     {"style": "attack",     "xg_range": [1.2, 1.9]},
    "Italy":       {"style": "defensive",  "xg_range": [0.9, 1.4]},
    "Colombia":    {"style": "balanced",   "xg_range": [1.0, 1.6]},
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ShotRequest(BaseModel):
    """
    What the caller must send in to ask 'what's the xG of this shot?' —
    where the shot was taken from, and a few yes/no facts about it
    (was it a header, a free kick, how many defenders were in the way).
    """
    x:                 float = Field(..., ge=0, le=120, description="Shot x-coordinate (StatsBomb: 0-120)")
    y:                 float = Field(..., ge=0, le=80,  description="Shot y-coordinate (StatsBomb: 0-80)")
    is_header:         int   = Field(0, ge=0, le=1)
    is_volley:         int   = Field(0, ge=0, le=1)
    is_free_kick:      int   = Field(0, ge=0, le=1)
    defenders_in_cone: int   = Field(0, ge=0, le=10)
    gk_set:            int   = Field(1, ge=0, le=1)


class ShotResponse(BaseModel):
    """
    What we hand back for a single shot: the xG number itself, plus the
    distance and angle we calculated along the way, so the caller can see
    the reasoning, not just the final probability.
    """
    xg:              float
    distance:        float
    angle_deg:       float
    is_big_chance:   bool
    centrality:      float


class MatchRequest(BaseModel):
    """
    What the caller sends in to compare two teams — just their names.
    """
    home_team: str = Field(..., description="Home team name (see /teams)")
    away_team: str = Field(..., description="Away team name (see /teams)")


class ShotLocation(BaseModel):
    """One simulated shot, in StatsBomb 120 x 80 pitch coordinates."""
    x: float
    y: float
    xg: float


class MatchResponse(BaseModel):
    """Match-level xG totals plus the individual shots used to make them."""
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_shots: int
    away_shots: int
    home_shot_locations: list[ShotLocation]
    away_shot_locations: list[ShotLocation]


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_derived(x: float, y: float) -> dict:
    """
    Takes a raw shot location on the pitch and works out the things that
    actually matter for scoring: how far from goal it is, and how wide
    the goal 'looks' from that spot (a shot from dead center sees almost
    the whole goal; a shot from a tight angle barely sees any of it).
    Also flags whether it counts as a clear-cut chance.
    """
    goal_x, goal_y = 120.0, 40.0
    dx = goal_x - x
    dy = goal_y - y
    distance = float(np.sqrt(dx**2 + dy**2))

    gp1 = np.array([120.0, 36.0])
    gp2 = np.array([120.0, 44.0])
    sloc = np.array([x, y])
    a = gp1 - sloc
    b = gp2 - sloc
    cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    angle = float(np.arccos(np.clip(cos_a, -1, 1)))

    centrality    = round(1.0 - abs(y - 40.0) / 40.0, 3)
    is_big_chance = int(distance < 12 and angle > 0.4)
    return {
        "distance":     round(distance, 3),
        "angle":        round(angle, 4),
        "centrality":   centrality,
        "is_big_chance": is_big_chance,
    }


def build_feature_vector(req: ShotRequest) -> np.ndarray:
    """
    Packages everything about one shot — distance, angle, header?,
    free kick?, defenders in the way, etc. — into the exact list of
    numbers, in the exact order, that the trained model expects to see.
    This has to match the training data's format exactly or the model's
    predictions won't make sense.
    """
    derived = compute_derived(req.x, req.y)
    vec = [
        derived["distance"],
        derived["angle"],
        derived["distance"] * derived["angle"],   # dist_angle
        req.is_header,
        req.is_volley,
        derived["is_big_chance"],
        req.is_free_kick,
        req.defenders_in_cone,
        req.gk_set,
        derived["centrality"],
    ]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def simulate_team_xg(team: str, side: str, n_shots: int) -> tuple[float, list[ShotLocation]]:
    """
    This project does not have event data for a future fixture, so it
    generates plausible shot locations. Every generated shot is scored
    by the trained XGBoost model. The exact same per-shot xG values are
    returned to the dashboard and added to the team's total.

    Important: the locations are simulated, not actual match events.
    """
    model = get_model()
    rng = random.Random()
    total_xg = 0.0
    shot_locations: list[ShotLocation] = []

    for _ in range(n_shots):
        if side == "home":
            x = max(80.0, min(118.0, rng.gauss(104.0, 8.0)))
        else:
            x = max(2.0, min(40.0, rng.gauss(16.0, 8.0)))
        y = max(10.0, min(70.0, rng.gauss(40.0, 9.0)))

        req = ShotRequest(
            x=x,
            y=y,
            is_header=1 if rng.random() < 0.18 else 0,
            is_volley=0,
            is_free_kick=1 if rng.random() < 0.08 else 0,
            defenders_in_cone=rng.randint(0, 3),
            gk_set=1,
        )
        fvec = build_feature_vector(req)
        model_xg = float(model.predict_proba(fvec)[0, 1])
        shot_xg = round(min(max(model_xg, 0.0), 0.95), 3)
        total_xg += shot_xg

        shot_locations.append(
            ShotLocation(x=round(x, 2), y=round(y, 2), xg=shot_xg)
        )

    return round(total_xg, 3), shot_locations


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    """
    A simple 'are you awake?' check. If this responds, the API is running.
    """
    return {"status": "ok", "service": "WC 2026 xG Engine", "version": "1.0.0"}


@app.get("/teams")
def list_teams():
    """
    Just hands back the static list of teams and their style profiles —
    no model, no calculation, purely a lookup.
    """
    return {
        "teams": [
            {"name": k, "style": v["style"], "xg_range": v["xg_range"]}
            for k, v in TEAMS.items()
        ],
    }


@app.get("/model/info")
def model_info():
    """
    Returns the model's stats report — how accurate it is, and which
    features (distance, angle, etc.) it relied on most. Lets someone
    inspect the model without needing to retrain it.
    """
    meta = get_meta()
    if not meta:
        return {"warning": "Model not trained yet. Run models/train.py."}
    return meta


@app.post("/predict/shot", response_model=ShotResponse)
def predict_shot(req: ShotRequest):
    """
    The main event: given one shot's details, work out distance and angle,
    feed everything into the trained model, and return the probability
    it results in a goal — the xG value.
    """
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    derived = compute_derived(req.x, req.y)
    fvec    = build_feature_vector(req)
    xg_val  = float(model.predict_proba(fvec)[0, 1])

    return ShotResponse(
        xg=round(xg_val, 4),
        distance=derived["distance"],
        angle_deg=round(float(np.degrees(derived["angle"])), 2),
        is_big_chance=bool(derived["is_big_chance"]),
        centrality=derived["centrality"],
    )


@app.post("/predict/match", response_model=MatchResponse)
def predict_match(req: MatchRequest):
    """
    Given two team names, checks they're valid and different, decides
    how many shots each team would plausibly take based on their style,
    simulates that many shots for each through the real model, and
    returns each team's total xG. This is the endpoint behind the
    Match Prediction tab.
    """
    if req.home_team not in TEAMS:
        raise HTTPException(status_code=400, detail=f"Unknown team: {req.home_team}. See /teams.")
    if req.away_team not in TEAMS:
        raise HTTPException(status_code=400, detail=f"Unknown team: {req.away_team}. See /teams.")
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must differ.")

    try:
        get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    style_shots = {"attack": 10, "balanced": 8, "possession": 9, "defensive": 6, "pressing": 7}
    h_n = style_shots.get(TEAMS[req.home_team]["style"], 8)
    a_n = style_shots.get(TEAMS[req.away_team]["style"], 8)

    home_xg, home_shot_locations = simulate_team_xg(req.home_team, "home", h_n)
    away_xg, away_shot_locations = simulate_team_xg(req.away_team, "away", a_n)

    return MatchResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        home_xg=home_xg,
        away_xg=away_xg,
        home_shots=h_n,
        away_shots=a_n,
        home_shot_locations=home_shot_locations,
        away_shot_locations=away_shot_locations,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)