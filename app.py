"""
WC 2026 xG Engine — Streamlit Dashboard
Run: streamlit run app.py
Requires API running at localhost:8000
"""

import streamlit as st
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WC 2026 xG Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "https://expected-goal-prediction-model.onrender.com"

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0A0F !important;
    color: #F0F0F0 !important;
}
[data-testid="stSidebar"] {
    background-color: #0D1510 !important;
    border-right: 1px solid #1A2E1E !important;
}
[data-testid="stSidebar"] * { color: #C0D0C4 !important; }

.main-title {
    font-family: 'Georgia', serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: #C9A84C;
    text-transform: uppercase;
    margin-bottom: 0;
}
.sub-title {
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #3A5A3E;
    text-transform: uppercase;
    margin-top: 0;
}
.metric-card {
    background: #0F1812;
    border: 1px solid #1A2A1E;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #3A5A3E;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #C9A84C;
    line-height: 1;
}
.metric-sub {
    font-size: 0.65rem;
    color: #253525;
    margin-top: 3px;
}
.section-label {
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    color: #3A5A3E;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 16px;
}
.team-display {
    text-align: center;
    padding: 12px;
    background: #0D1510;
    border: 1px solid #1A2A1E;
    border-radius: 8px;
}
.team-name-big {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #EEE;
}
.team-style-badge {
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    color: #3A5A3E;
    text-transform: uppercase;
}
.stSelectbox > div > div {
    background-color: #111A13 !important;
    border: 1px solid #1E2E22 !important;
    color: #DDD !important;
    border-radius: 6px !important;
}
.stButton > button {
    background-color: #C9A84C !important;
    color: #0A0A0F !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: #1A2A1E !important; }
.stTabs [data-baseweb="tab-list"] {
    background-color: #0D1510 !important;
    border-bottom: 1px solid #1A2A1E !important;
}
.stTabs [data-baseweb="tab"] {
    color: #4A6A50 !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: #C9A84C !important;
    border-bottom: 2px solid #C9A84C !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEAMS = [
    "Argentina", "Brazil", "France", "England", "Spain",
    "Germany", "Portugal", "Netherlands", "Morocco", "USA",
    "Mexico", "Canada", "Japan", "South Korea", "Senegal",
    "Uruguay", "Croatia", "Belgium", "Italy", "Colombia",
]

VENUES = [
    "MetLife Stadium, NJ", "AT&T Stadium, TX", "SoFi Stadium, CA",
    "Hard Rock Stadium, FL", "Levi's Stadium, CA", "Arrowhead Stadium, MO",
    "Lincoln Financial Field, PA", "Gillette Stadium, MA", "Estadio Azteca, MX",
    "Estadio Akron, MX", "BC Place, Canada", "BMO Field, Canada",
]

STAGES = ["Group Stage", "Round of 32", "Round of 16", "Quarter-Final", "Semi-Final", "Final"]


def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def check_api() -> bool:
    result = api_get("/")
    return result is not None and result.get("status") == "ok"


# ---------------------------------------------------------------------------
# Pitch drawing helpers
# ---------------------------------------------------------------------------

def draw_full_pitch_with_shots(home_shots, away_shots, home_team, away_team):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0A0A10")

    for ax, shots, team, side in [
        (axes[0], home_shots, home_team, "home"),
        (axes[1], away_shots, away_team, "away"),
    ]:
        ax.set_facecolor("#0C1A10")
        ax.set_xlim(-2, 122)
        ax.set_ylim(-2, 82)
        ax.set_aspect("equal")
        ax.axis("off")

        # Pitch outline
        pitch = patches.Rectangle((0, 0), 120, 80,
                                   linewidth=1, edgecolor="#1A3A20", facecolor="none")
        ax.add_patch(pitch)

        # Centre circle + spot
        centre = plt.Circle((60, 40), 10, linewidth=0.8,
                             edgecolor="#1A3A20", facecolor="none")
        ax.add_patch(centre)
        ax.plot(60, 40, "o", color="#1A3A20", markersize=3)
        ax.axvline(60, color="#1A3A20", linewidth=0.8)

        # Both penalty boxes
        for bx, is_right in [(0, False), (102, True)]:
            box = patches.Rectangle((bx, 18), 18, 44,
                                     linewidth=0.8, edgecolor="#1A3A20", facecolor="none")
            ax.add_patch(box)
            six = patches.Rectangle((bx if not is_right else 114, 30), 6, 20,
                                     linewidth=0.8, edgecolor="#162A18", facecolor="none")
            ax.add_patch(six)

        # Goals
        for gx in [(-2, True), (120, False)]:
            goal = patches.Rectangle((gx[0], 36), 2, 8,
                                      linewidth=1, edgecolor="#C9A84C66", facecolor="#C9A84C11")
            ax.add_patch(goal)

        # Penalty spots
        ax.plot(12, 40, "o", color="#1A3A20", markersize=2)
        ax.plot(108, 40, "o", color="#1A3A20", markersize=2)

        # Shot scatter
        if shots:
            for shot in shots:
                try:
                    raw_x = float(shot["x"])
                    raw_y = float(shot["y"])
                    xg = min(max(float(shot["xg"]), 0.0), 1.0)
                except (KeyError, TypeError, ValueError):
                    continue

                # Mirror away-team x coordinates so both panels show attacks
                # towards the right-hand goal.
                plot_x = raw_x if side == "home" else 120.0 - raw_x
                size = 60 + xg * 400
                color = "#C9A84C" if xg >= 0.15 else "#2A6A3A"
                alpha = 0.55 + xg * 0.4
                ax.scatter(
                    plot_x, raw_y, s=size, c=color,
                    edgecolors="#E8D9A8" if xg >= 0.15 else "#6C9B75",
                    linewidths=0.8, alpha=alpha, zorder=5,
                )

        ax.set_title(team.upper(), color="#C9A84C",
                     fontsize=9,
                     pad=6, fontweight="bold")
        ax.text(60, -1.5, team.upper(), color="#C9A84C",
                fontsize=9, ha="center", va="top", fontweight="bold",
                transform=ax.transData)

    plt.tight_layout(pad=0.5)
    return fig


def draw_shot_map_single(x, y, xg_val):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0C1410")
    ax.set_facecolor("#0C1410")

    half = patches.Rectangle((60, 0), 60, 80,
                               linewidth=1, edgecolor="#1A3A20", facecolor="#0C1A10")
    ax.add_patch(half)
    box = patches.Rectangle((102, 18), 18, 44,
                              linewidth=0.8, edgecolor="#1A3A20", facecolor="none")
    ax.add_patch(box)
    six = patches.Rectangle((114, 30), 6, 20,
                              linewidth=0.8, edgecolor="#162A18", facecolor="none")
    ax.add_patch(six)
    goal = patches.Rectangle((120, 36), 2, 8,
                               linewidth=1, edgecolor="#C9A84C66", facecolor="#C9A84C11")
    ax.add_patch(goal)
    ax.plot(108, 40, "o", color="#1A3A20", markersize=2)

    size = 80 + xg_val * 300
    ax.scatter(x, y, s=size, c="#C9A84C", edgecolors="#C9A84C",
               linewidths=1.5, alpha=0.9, zorder=5)
    ax.annotate(f"xG = {xg_val:.3f}",
                xy=(x, y), xytext=(x - 8, y + 5),
                color="#C9A84C", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C9A84C44"))

    ax.set_xlim(58, 123)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


def draw_shap_bar(shap_importance: dict):
    features = list(shap_importance.keys())
    values   = list(shap_importance.values())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0F1812")
    ax.set_facecolor("#0F1812")

    colors = ["#C9A84C" if v == max(values) else "#2A4A2E" for v in values]
    bars   = ax.barh(features, values, color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#6A8A6E", fontsize=7)

    ax.set_xlabel("Mean |SHAP|", color="#3A5A3E", fontsize=8)
    ax.tick_params(colors="#4A6A50", labelsize=8)
    ax.spines[:].set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="main-title">xG ENGINE</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">FIFA World Cup 2026</p>', unsafe_allow_html=True)
    st.markdown("---")

    api_ok = check_api()
    if api_ok:
        st.success("API connected", icon=None)
    else:
        st.error("API offline — start uvicorn")
        st.code("cd api\nuvicorn main:app --port 8000")

    st.markdown("---")
    st.markdown('<p class="section-label">Match Setup</p>', unsafe_allow_html=True)
    home_team   = st.selectbox("Home Team", TEAMS, index=3)
    away_team   = st.selectbox("Away Team", TEAMS, index=1)
    venue       = st.selectbox("Venue", VENUES, index=1)
    stage       = st.selectbox("Stage", STAGES, index=0)
    neutral     = st.checkbox("Neutral venue (no home advantage)", value=True)
    predict_btn = st.button("Run Prediction")

    st.markdown("---")
    st.markdown('<p class="section-label">Single Shot xG</p>', unsafe_allow_html=True)
    sx        = st.slider("Shot X (yards)", 60.0, 119.0, 105.0, 0.5)
    sy        = st.slider("Shot Y (yards)", 0.0, 80.0, 40.0, 0.5)
    is_header = st.checkbox("Header")
    is_fk     = st.checkbox("Free Kick")
    defenders = st.slider("Defenders in cone", 0, 5, 1)
    shot_btn  = st.button("Calculate Shot xG")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<p class="main-title">WC 2026 — xG ENGINE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Expected Goals · StatsBomb · XGBoost · AUC 0.86</p>',
            unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["MATCH PREDICTION", "SHOT CALCULATOR", "MODEL INFO"])


# ── Tab 1: Match Prediction ──────────────────────────────────────────────────
with tab1:
    if predict_btn:
        if home_team == away_team:
            st.warning("Select two different teams.")
        else:
            with st.spinner("Running xG model..."):
                result = api_post("/predict/match", {
                    "home_team":     home_team,
                    "away_team":     away_team,
                    "venue":         venue,
                    "stage":         stage,
                    "neutral_venue": neutral,
                })
            if result:
                st.session_state["match_result"] = result
            elif not api_ok:
                st.error("API is not running. Start uvicorn first.")

    if "match_result" in st.session_state:
        r = st.session_state["match_result"]

        # Teams header
        c1, c2, c3 = st.columns([5, 2, 5])
        with c1:
            st.markdown(f"""
            <div class="team-display">
                <div class="team-name-big">{r['home_team']}</div>
                <div class="team-style-badge">Home</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="text-align:center;padding-top:16px;">
                <div style="font-size:1.1rem;letter-spacing:3px;color:#3A5A3E;font-weight:700;">VS</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="team-display">
                <div class="team-name-big">{r['away_team']}</div>
                <div class="team-style-badge">Away</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # xG + shots + simulated score
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{r['home_team']} xG</div>
                <div class="metric-value">{r['home_xg']:.2f}</div>
                <div class="metric-sub">Expected Goals</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{r['away_team']} xG</div>
                <div class="metric-value">{r['away_xg']:.2f}</div>
                <div class="metric-sub">Expected Goals</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Shots</div>
                <div class="metric-value">{r['home_shots']} – {r['away_shots']}</div>
                <div class="metric-sub">Home – Away</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            if 'home_simulated_score' in r:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Simulated Score</div>
                    <div class="metric-value">{r['home_simulated_score']} – {r['away_simulated_score']}</div>
                    <div class="metric-sub">Most likely simulation</div>
                </div>""", unsafe_allow_html=True)

        # Win probability bar
        if "home_win_prob" in r:
            st.markdown('<p class="section-label">WIN PROBABILITY</p>', unsafe_allow_html=True)
            hw = r["home_win_prob"]
            dr = r["draw_prob"]
            aw = r["away_win_prob"]
            bar_html = f"""
            <div style="display:flex;width:100%;border-radius:6px;overflow:hidden;height:36px;margin-bottom:4px;">
                <div style="width:{hw*100:.0f}%;background:#C9A84C;display:flex;align-items:center;justify-content:center;">
                    <span style="font-size:0.75rem;font-weight:700;color:#0A0A0F;">{hw*100:.0f}%</span>
                </div>
                <div style="width:{dr*100:.0f}%;background:#2A4A2E;display:flex;align-items:center;justify-content:center;">
                    <span style="font-size:0.75rem;font-weight:700;color:#C0D0C4;">{dr*100:.0f}%</span>
                </div>
                <div style="width:{aw*100:.0f}%;background:#1D6A50;display:flex;align-items:center;justify-content:center;">
                    <span style="font-size:0.75rem;font-weight:700;color:#EEE;">{aw*100:.0f}%</span>
                </div>
            </div>
            <div style="display:flex;width:100%;justify-content:space-between;">
                <span style="font-size:0.6rem;color:#3A5A3E;letter-spacing:1.5px;text-transform:uppercase;">{r['home_team']} WIN</span>
                <span style="font-size:0.6rem;color:#3A5A3E;letter-spacing:1.5px;text-transform:uppercase;">DRAW</span>
                <span style="font-size:0.6rem;color:#3A5A3E;letter-spacing:1.5px;text-transform:uppercase;">{r['away_team']} WIN</span>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        # Shot map — generated by the match API from the same shot-level xG
        # predictions that are summed into the team totals above.
        # These are simulated locations for a projected fixture, not actual
        # event-data shots from a match that has already been played.
        home_shots = r.get("home_shot_locations") or []
        away_shots = r.get("away_shot_locations") or []
        st.markdown('<p class="section-label">SHOT MAP</p>', unsafe_allow_html=True)
        if home_shots or away_shots:
            map_col1, map_col2 = st.columns(2)
            with map_col1:
                st.caption(f"{r['home_team']} · {len(home_shots)} simulated shots")
            with map_col2:
                st.caption(f"{r['away_team']} · {len(away_shots)} simulated shots")
            fig = draw_full_pitch_with_shots(
                home_shots, away_shots, r["home_team"], r["away_team"]
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Each marker represents a simulated shot. Larger markers indicate higher xG; "
                "gold = xG ≥ 0.15, green = xG < 0.15."
            )
        else:
            st.warning(
                "No shot coordinates were returned. Restart the updated FastAPI backend "
                "and click Run Prediction again."
            )

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#2A4A2E;">
            <div style="font-size:3rem;margin-bottom:16px;">—</div>
            <div style="font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;">
                Select two teams in the sidebar and click Run Prediction
            </div>
        </div>""", unsafe_allow_html=True)


# ── Tab 2: Shot Calculator ───────────────────────────────────────────────────
with tab2:
    if shot_btn:
        with st.spinner("Calculating xG..."):
            result = api_post("/predict/shot", {
                "x":                 sx,
                "y":                 sy,
                "is_header":         int(is_header),
                "is_volley":         0,
                "is_free_kick":      int(is_fk),
                "defenders_in_cone": defenders,
                "gk_set":            1,
            })

        if result:
            st.session_state["shot_result"] = result
            st.session_state["shot_xy"]     = (sx, sy)

    if "shot_result" in st.session_state:
        sr   = st.session_state["shot_result"]
        s_xy = st.session_state["shot_xy"]

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = draw_shot_map_single(s_xy[0], s_xy[1], sr["xg"])
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Goal</div>
                <div class="metric-value">{sr['xg']:.4f}</div>
                <div class="metric-sub">Probability of scoring</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Distance</div>
                <div class="metric-value" style="font-size:1.4rem">{sr['distance']:.1f} yd</div>
                <div class="metric-sub">From goal centre</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Angle</div>
                <div class="metric-value" style="font-size:1.4rem">{sr['angle_deg']:.1f} deg</div>
                <div class="metric-sub">Goal mouth angle</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Big Chance</div>
                <div class="metric-value" style="font-size:1.4rem;color:{'#1D9E75' if sr['is_big_chance'] else '#3A5A3E'}">
                    {'YES' if sr['is_big_chance'] else 'NO'}
                </div>
                <div class="metric-sub">dist &lt; 12yd and angle &gt; 23 deg</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#2A4A2E;">
            <div style="font-size:3rem;margin-bottom:16px;">—</div>
            <div style="font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;">
                Set shot coordinates in the sidebar and click Calculate Shot xG
            </div>
        </div>""", unsafe_allow_html=True)


# ── Tab 3: Model Info ────────────────────────────────────────────────────────
with tab3:
    meta = api_get("/model/info")
    if meta and "warning" not in meta:
        c1, c2, c3, c4 = st.columns(4)
        cv  = meta.get("cv_results", {})
        trn = meta.get("train_metrics", {})

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CV AUC-ROC</div>
                <div class="metric-value">{cv.get('auc_mean', 0):.3f}</div>
                <div class="metric-sub">5-fold stratified</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Log Loss</div>
                <div class="metric-value">{cv.get('logloss_mean', 0):.3f}</div>
                <div class="metric-sub">Cross-validation</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Full AUC</div>
                <div class="metric-value">{trn.get('auc_roc', 0):.3f}</div>
                <div class="metric-sub">Train set</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Brier Score</div>
                <div class="metric-value">{trn.get('brier', 0):.3f}</div>
                <div class="metric-sub">Calibration</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-label">SHAP Feature Importance</p>',
                    unsafe_allow_html=True)
        shap_imp = meta.get("shap_importance", {})
        if shap_imp:
            c1, c2 = st.columns([2, 3])
            with c1:
                fig = draw_shap_bar(shap_imp)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            with c2:
                import pandas as pd
                df_shap = pd.DataFrame([
                    {"Feature": k, "SHAP Importance": round(v, 4)}
                    for k, v in shap_imp.items()
                ])
                st.dataframe(df_shap, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-label">Model Parameters</p>',
                    unsafe_allow_html=True)
        params = meta.get("model_params", {})
        if params:
            import pandas as pd
            df_params = pd.DataFrame([
                {"Parameter": k, "Value": str(v)} for k, v in params.items()
            ])
            st.dataframe(df_params, use_container_width=True, hide_index=True)
    else:
        st.error("Model metadata not found. Run models/train.py first.")