"""
FastAPI xG Engine — WC 2026
Endpoints:
  GET  /                      health check
  POST /predict/shot          xG for a single shot
  POST /predict/match         simulated match xG for two teams
  GET  /model/info            model metadata + SHAP importances
  GET  /teams                 list of WC 2026 teams with style profiles
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
    "gk_set", "centrality","dist_angle"
]


def get_model() -> xgb.XGBClassifier:
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
    global _meta
    if _meta is None and os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta = json.load(f)
    return _meta or {}


# ---------------------------------------------------------------------------
# Team profiles
# ---------------------------------------------------------------------------

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

WC2026_VENUES = [
    "MetLife Stadium, NJ",
    "Rose Bowl, LA",
    "AT&T Stadium, TX",
    "Levi's Stadium, CA",
    "Arrowhead Stadium, KC",
    "Hard Rock Stadium, MIA",
    "Lincoln Financial Field, PHI",
    "SoFi Stadium, LA",
    "Estadio Azteca, Mexico City",
    "Estadio BBVA, Monterrey",
    "BMO Field, Toronto",
    "BC Place, Vancouver",
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ShotRequest(BaseModel):
    x:                 float = Field(..., ge=0, le=120, description="Shot x-coordinate (StatsBomb: 0-120)")
    y:                 float = Field(..., ge=0, le=80,  description="Shot y-coordinate (StatsBomb: 0-80)")
    is_header:         int   = Field(0, ge=0, le=1)
    is_volley:         int   = Field(0, ge=0, le=1)
    is_free_kick:      int   = Field(0, ge=0, le=1)
    defenders_in_cone: int   = Field(0, ge=0, le=10)
    gk_set:            int   = Field(1, ge=0, le=1)


class ShotResponse(BaseModel):
    xg:              float
    distance:        float
    angle_deg:       float
    is_big_chance:   bool
    centrality:      float


class MatchRequest(BaseModel):
    home_team: str = Field(..., description="Home team name (see /teams)")
    away_team: str = Field(..., description="Away team name (see /teams)")
    venue:     str = Field("MetLife Stadium, NJ")
    stage:     str = Field("Group Stage")


class ShotSim(BaseModel):
    x:      float
    y:      float
    xg:     float
    goal:   bool
    side:   str


class MatchResponse(BaseModel):
    home_team:      str
    away_team:      str
    venue:          str
    home_xg:        float
    away_xg:        float
    home_win_prob:  float
    draw_prob:      float
    away_win_prob:  float
    home_shots:     list[ShotSim]
    away_shots:     list[ShotSim]
    model_used:     str


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_derived(x: float, y: float) -> dict:
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


def simulate_team_shots(team: str, side: str, n_shots: int) -> tuple[list[ShotSim], float]:
    """
    Generate realistic shot positions for a team based on xG range profile,
    then run each through the model.
    """
    model   = get_model()
    profile = TEAMS[team]
    lo, hi  = profile["xg_range"]
    shots   = []
    total_xg = 0.0

    rng = random.Random()

    for _ in range(n_shots):
        # Shots cluster around the box (x: 90-118, y: 28-52) with tails
        if side == "home":
            x = rng.gauss(104, 8)
            x = max(80, min(118, x))
        else:
            x = rng.gauss(16, 8)
            x = max(2, min(40, x))

        y = rng.gauss(40, 9)
        y = max(10, min(70, y))

        defenders = rng.randint(0, 3)
        is_header = 1 if rng.random() < 0.18 else 0
        is_fk     = 1 if rng.random() < 0.08 else 0

        req = ShotRequest(
            x=x, y=y,
            is_header=is_header,
            is_volley=0,
            is_free_kick=is_fk,
            defenders_in_cone=defenders,
            gk_set=1,
        )
        fvec = build_feature_vector(req)
        xg_val = float(model.predict_proba(fvec)[0, 1])
        xg_val = round(min(xg_val, 0.95), 3)
        total_xg += xg_val

        is_goal = rng.random() < xg_val
        shots.append(ShotSim(
            x=round(x, 1), y=round(y, 1),
            xg=xg_val, goal=is_goal, side=side,
        ))

    return shots, round(total_xg, 3)


def compute_win_probs(xg_h: float, xg_a: float) -> tuple[float, float, float]:
    diff  = xg_h - xg_a
    p_h   = max(0.10, min(0.70, 0.38 + diff * 0.10))
    p_a   = max(0.10, min(0.70, 0.38 - diff * 0.10))
    p_d   = max(0.05, 1.0 - p_h - p_a)
    total = p_h + p_d + p_a
    return round(p_h / total, 3), round(p_d / total, 3), round(p_a / total, 3)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "service": "WC 2026 xG Engine", "version": "1.0.0"}


@app.get("/teams")
def list_teams():
    return {
        "teams": [
            {"name": k, "style": v["style"], "xg_range": v["xg_range"]}
            for k, v in TEAMS.items()
        ],
        "venues": WC2026_VENUES,
    }


@app.get("/model/info")
def model_info():
    meta = get_meta()
    if not meta:
        return {"warning": "Model not trained yet. Run models/train.py."}
    return meta


@app.post("/predict/shot", response_model=ShotResponse)
def predict_shot(req: ShotRequest):
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

    h_profile = TEAMS[req.home_team]
    a_profile = TEAMS[req.away_team]

    # Shot count scales with team style
    style_shots = {"attack": 10, "balanced": 8, "possession": 9, "defensive": 6, "pressing": 7}
    h_n = style_shots.get(h_profile["style"], 11)
    a_n = style_shots.get(a_profile["style"], 11)

    home_shots, home_xg = simulate_team_shots(req.home_team, "home", h_n)
    away_shots, away_xg = simulate_team_shots(req.away_team, "away", a_n)
    ph, pd, pa           = compute_win_probs(home_xg, away_xg)

    return MatchResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        venue=req.venue,
        home_xg=home_xg,
        away_xg=away_xg,
        home_win_prob=ph,
        draw_prob=pd,
        away_win_prob=pa,
        home_shots=home_shots,
        away_shots=away_shots,
        model_used="xgboost_statsbomb_v1",
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
