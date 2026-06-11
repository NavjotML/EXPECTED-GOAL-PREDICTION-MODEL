"""
StatsBomb data extraction -- shot-level features for xG model.
Pulls from StatsBomb open data: World Cup 2022 (competition_id=43, season_id=106)
and additional competitions for training volume.
"""

import pandas as pd
import numpy as np
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")


COMPETITIONS = [
    # International tournaments — most relevant
    {"competition_id": 43, "season_id": 106, "label": "WC 2022"},
    {"competition_id": 43, "season_id":   3, "label": "WC 2018"},
    {"competition_id": 55, "season_id":  43, "label": "Euro 2020"},
    {"competition_id": 55, "season_id": 282, "label": "Euro 2024"},
    {"competition_id": 53, "season_id":  44, "label": "Copa America 2021"},
    {"competition_id": 223,"season_id": 282, "label": "Copa America 2024"},
    {"competition_id": 1267,"season_id":107, "label": "AFCON 2023"},

    # Club football — volume
    {"competition_id": 11, "season_id":  90, "label": "La Liga 2020/21"},
    {"competition_id": 11, "season_id":  42, "label": "La Liga 2019/20"},
    {"competition_id": 16, "season_id":   4, "label": "UCL 2018/19"},
    {"competition_id": 16, "season_id":   1, "label": "UCL 2017/18"},
    {"competition_id":  2, "season_id":  27, "label": "Premier League 2015/16"},
    {"competition_id":  7, "season_id": 108, "label": "Ligue 1 2021/22"},
]

def parse_field(val):
    """Safely extract name from either a dict {'name': ...} or a plain string."""
    if isinstance(val, dict):
        return val.get("name", "")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)


def extract_shot_features(event: dict):
    loc = event.get("location")
    if not loc or len(loc) < 2:
        return None

    x, y = float(loc[0]), float(loc[1])

    goal_x, goal_y = 120.0, 40.0
    dx = goal_x - x
    dy = goal_y - y
    distance = float(np.sqrt(dx**2 + dy**2))

    gp1  = np.array([120.0, 36.0])
    gp2  = np.array([120.0, 44.0])
    sloc = np.array([x, y])
    a    = gp1 - sloc
    b    = gp2 - sloc
    cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    angle = float(np.arccos(np.clip(cos_a, -1, 1)))

    outcome_name   = parse_field(event.get("shot_outcome"))
    shot_type_name = parse_field(event.get("shot_type"))
    body_part_name = parse_field(event.get("shot_body_part"))
    technique_name = parse_field(event.get("shot_technique"))

    freeze_frame      = event.get("shot_freeze_frame", [])
    defenders_in_cone = 0
    gk_present        = False

    if isinstance(freeze_frame, list):
        for player in freeze_frame:
            if not isinstance(player, dict):
                continue
            ploc = player.get("location", [])
            if len(ploc) < 2:
                continue
            px, py = float(ploc[0]), float(ploc[1])
            if player.get("teammate", False):
                continue
            pos = parse_field(player.get("position", ""))
            if pos == "Goalkeeper":
                gk_present = True
                continue
            if px > x and abs(py - 40) < 6:
                defenders_in_cone += 1

    is_goal       = 1 if outcome_name == "Goal" else 0
    is_header     = 1 if body_part_name == "Head" else 0
    is_volley     = 1 if "Volley" in technique_name else 0
    is_penalty    = 1 if shot_type_name == "Penalty" else 0
    is_free_kick  = 1 if shot_type_name == "Free Kick" else 0
    is_big_chance = 1 if (shot_type_name == "Open Play" and distance < 12 and angle > 0.4) else 0
    gk_set        = 1 if gk_present else 0
    centrality    = round(1.0 - abs(y - 40.0) / 40.0, 3)

    return {
        "x":                  round(x, 2),
        "y":                  round(y, 2),
        "distance":           round(distance, 3),
        "angle":              round(angle, 4),
        "is_header":          is_header,
        "is_volley":          is_volley,
        "is_big_chance":      is_big_chance,
        "is_penalty":         is_penalty,
        "is_free_kick":       is_free_kick,
        "defenders_in_cone":  defenders_in_cone,
        "gk_set":             gk_set,
        "centrality":         centrality,
        "goal":               is_goal,
        "_outcome_raw":       outcome_name,  
        "dist_angle":  round(distance * angle, 3), # kept for debug, dropped before training
    }


def load_competition_shots(competition_id: int, season_id: int, label: str) -> pd.DataFrame:
    print(f"  Loading {label}...")
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        print(f"  Failed to load matches for {label}: {e}")
        return pd.DataFrame()

    rows        = []
    first_match = True

    for _, match in matches.iterrows():
        try:
            events   = sb.events(match_id=match["match_id"])
            type_col = events["type"]

            # statsbombpy 1.17 returns plain strings; older versions return dicts
            if len(type_col) > 0 and isinstance(type_col.iloc[0], dict):
                shots = events[type_col.apply(lambda t: t.get("name", "")) == "Shot"]
            else:
                shots = events[type_col == "Shot"]

            # Debug print on very first match of first competition
            if first_match and len(shots) > 0:
                first_match = False
                s0 = shots.iloc[0]
                print(f"    [debug] outcome  raw: {repr(s0.get('shot_outcome'))}")
                print(f"    [debug] type     raw: {repr(s0.get('shot_type'))}")
                print(f"    [debug] bodypart raw: {repr(s0.get('shot_body_part'))}")

            for _, shot in shots.iterrows():
                feat = extract_shot_features(shot.to_dict())
                if feat:
                    feat["match_id"]    = match["match_id"]
                    feat["competition"] = label
                    rows.append(feat)

        except Exception as e:
            print(f"    [warn] match {match['match_id']}: {e}")
            continue

    goals = sum(1 for r in rows if r.get("goal") == 1)
    print(f"    -> {len(rows)} shots | {goals} goals ({goals / max(len(rows), 1) * 100:.1f}%)")
    return pd.DataFrame(rows)


def build_dataset(output_path: str = "data/shots.csv") -> pd.DataFrame:
    print("Building shot dataset from StatsBomb open data...")
    all_shots = []

    for comp in COMPETITIONS:
        df = load_competition_shots(**comp)
        if not df.empty:
            all_shots.append(df)

    if not all_shots:
        raise RuntimeError("No shot data extracted.")

    full = pd.concat(all_shots, ignore_index=True)

    # Drop debug column before saving
    if "_outcome_raw" in full.columns:
        full = full.drop(columns=["_outcome_raw"])

    full = full.dropna(subset=["distance", "angle", "goal"])
    non_pen = full[full["is_penalty"] == 0].copy()

    print(f"\nTotal shots:       {len(full)}")
    print(f"Non-penalty shots: {len(non_pen)}")
    print(f"Goals:             {int(non_pen['goal'].sum())}")
    print(f"Goal rate:         {non_pen['goal'].mean():.3f}")

    full.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return non_pen


if __name__ == "__main__":
    build_dataset()