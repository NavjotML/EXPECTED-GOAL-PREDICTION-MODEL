# WC 2026 xG Engine

Expected Goals (xG) prediction system for FIFA World Cup 2026.  
Trained on StatsBomb open data using XGBoost with full SHAP explainability,  
served via a FastAPI backend.

---

## Architecture

```
data/
  extract.py          StatsBomb open data pull — shot-level feature engineering
  shots.csv           Generated dataset (15k+ shots across competitions)

models/
  train.py            XGBoost training — 5-fold CV, calibration, SHAP
  explain.py          Standalone SHAP deep dive — waterfall, beeswarm, dependence
  xg_model.json       Trained model (produced by train.py)
  model_meta.json     CV results, metrics, feature importances
  shap_plots/         PNG outputs from explain.py

api/
  main.py             FastAPI application
                        POST /predict/shot    single shot xG
                        POST /predict/match   full match simulation
                        GET  /teams           WC 2026 team profiles
                        GET  /model/info      metadata + SHAP importances

dashboard/            (frontend widget — see xg_engine UI)
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Extract shot data

```bash
python data/extract.py
```

Pulls shot events from StatsBomb open data across 4 competitions (~15k shots).  
Saves to `data/shots.csv`.

### 3. Train the model

```bash
python models/train.py
```

Runs 5-fold stratified CV, trains on full dataset, outputs:
- `models/xg_model.json`
- `models/model_meta.json`
- `models/shap_plots/`
- `models/calibration.png`

Expected metrics (StatsBomb open data):

| Metric     | Value  |
|------------|--------|
| AUC-ROC    | ~0.79  |
| Log loss   | ~0.24  |
| Brier score| ~0.07  |

### 4. Run SHAP explainability (optional)

```bash
python models/explain.py
```

Generates waterfall plots for best/worst/median chances, beeswarm distribution,
and dependence plots for distance, angle, and centrality.

### 5. Start the API

```bash
cd api
uvicorn main:app --reload --port 8000
```

API docs auto-generated at: http://localhost:8000/docs

---

## API Usage

### Single shot xG

```bash
curl -X POST http://localhost:8000/predict/shot \
  -H "Content-Type: application/json" \
  -d '{
    "x": 108,
    "y": 38,
    "is_header": 0,
    "is_volley": 0,
    "is_free_kick": 0,
    "defenders_in_cone": 1,
    "gk_set": 1
  }'
```

Response:
```json
{
  "xg": 0.3471,
  "distance": 12.04,
  "angle_deg": 28.6,
  "is_big_chance": true,
  "centrality": 0.95
}
```

### Full match prediction

```bash
curl -X POST http://localhost:8000/predict/match \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "France",
    "venue": "MetLife Stadium, NJ",
    "stage": "Quarter-Final"
  }'
```

Response includes:
- `home_xg`, `away_xg` — summed expected goals
- `home_win_prob`, `draw_prob`, `away_win_prob`
- `home_shots`, `away_shots` — per-shot breakdown with x, y, xG, and goal simulation

---

## Features

| Feature            | Description                                          |
|--------------------|------------------------------------------------------|
| `distance`         | Yards from centre of goal                            |
| `angle`            | Angle subtended by goal mouth (radians)              |
| `is_header`        | 1 if headed shot                                     |
| `is_volley`        | 1 if volley                                          |
| `is_big_chance`    | 1 if distance < 12 yd and angle > 0.4 rad            |
| `is_free_kick`     | 1 if direct free kick                                |
| `defenders_in_cone`| Number of defenders between shot and goal            |
| `gk_set`           | 1 if goalkeeper is in position                       |
| `centrality`       | Lateral centrality — 1.0 = perfectly central         |

---

## Model Details

- Algorithm: XGBoost classifier (`scale_pos_weight=8` for class imbalance)
- Training data: WC 2022 + La Liga 19/20, 20/21 + UCL 18/19 (StatsBomb open)
- Validation: 5-fold stratified cross-validation
- Calibration checked via calibration curve plot
- Penalties excluded from main model (constant xG ~0.76 used separately)

---

## Extending for Heat Stress

The xG model feeds into the broader WC 2026 heat stress system:

```python
# In the heat stress module:
base_xg = predict_xg(shot_features)
heat_factor = compute_heat_degradation(wbgt_index, minute, position)
adjusted_xg = base_xg * heat_factor   # xG degrades in the final 20 min under heat
```

See `heat_stress/` module (next phase) for WBGT integration and press-intensity modeling.

---

## Data Sources

- **StatsBomb open data**: https://github.com/statsbomb/open-data
- **statsbombpy**: https://github.com/statsbomb/statsbombpy
- **WC 2026 venue data**: FIFA official

---

## Project

Built as portfolio project targeting IMPECT / Catapult Sports internship applications.  
Author: Navjot Singh | github.com/NavjotML
