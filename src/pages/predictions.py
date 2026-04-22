"""
pages/predictions.py — Race Predictions Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.ui_components import section_header, position_badge, team_badge
from src.utils import get_team_color
from src.feature_engineering import FEATURE_COLUMNS


def render(df, df_featured, models, results):
    """Render the Race Predictions page."""
    section_header("Race Predictions", subtitle="ML-powered finishing order predictions")

    if not models or "scaler_reg" not in models:
        st.warning("Models not loaded. Please ensure models are trained first.")
        return

    # Race selector
    years = sorted(df["year"].unique(), reverse=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        sel_year = st.selectbox("Select Season", years, key="pred_year")
    
    year_races = df[df["year"] == sel_year][["raceId", "race_name", "round"]].drop_duplicates()
    year_races = year_races.sort_values("round")
    race_options = {f"R{r} — {n}": rid for _, (rid, n, r) in year_races.iterrows()}
    
    with col2:
        sel_race_label = st.selectbox("Select Race", list(race_options.keys()), key="pred_race")
    
    if not sel_race_label:
        return
    
    sel_race_id = race_options[sel_race_label]
    race_data = df[df["raceId"] == sel_race_id].copy()
    race_feat = df_featured[df_featured["raceId"] == sel_race_id].copy()
    
    if race_data.empty:
        st.info("No data available for this race.")
        return

    st.markdown("---")

    # Get predictions
    tab1, tab2 = st.tabs(["Predicted Order", "Win Probability"])
    
    with tab1:
        _render_predicted_order(race_data, race_feat, models)
    
    with tab2:
        _render_win_probability(race_data, race_feat, models)


def _render_predicted_order(race_data, race_feat, models):
    """Render predicted finishing order vs actual."""
    if race_feat.empty or "reg_XGBoost" not in models:
        st.info("Feature data not available for this race.")
        return

    scaler = models.get("scaler_reg")
    reg_model = models.get("reg_XGBoost", models.get("reg_Random Forest"))
    clf_model = models.get("clf_Random Forest", models.get("clf_XGBoost"))
    scaler_clf = models.get("scaler_clf")

    # Prepare features
    feature_cols_available = [c for c in FEATURE_COLUMNS if c in race_feat.columns]
    X = race_feat[feature_cols_available].copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]
    X = X.fillna(0)

    # Predict positions
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    predicted_pos = reg_model.predict(X_scaled)

    # Predict win probability
    win_prob = np.zeros(len(X))
    if clf_model and scaler_clf:
        X_clf_scaled = pd.DataFrame(scaler_clf.transform(X), columns=X.columns, index=X.index)
        win_prob = clf_model.predict_proba(X_clf_scaled)[:, 1]

    # Build results table
    pred_df = race_data[["driver_name", "constructor_name", "grid", "positionOrder"]].copy()
    pred_df = pred_df.iloc[:len(predicted_pos)]
    pred_df["predicted_pos"] = np.round(predicted_pos, 1)
    pred_df["win_prob"] = (win_prob * 100).round(1)
    pred_df["pred_rank"] = pred_df["predicted_pos"].rank(method="min").astype(int)
    pred_df = pred_df.sort_values("pred_rank")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ML Predicted Order")
        for i, (_, row) in enumerate(pred_df.head(10).iterrows()):
            color = get_team_color(row["constructor_name"])
            pos_html = position_badge(i + 1)
            team_html = team_badge(row["constructor_name"], color)
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:8px 12px; margin:4px 0;
                        background:rgba(22,27,34,0.8); border-radius:8px; border-left:3px solid {color};">
                <div style="min-width:50px;">{pos_html}</div>
                <div style="flex:1; margin-left:10px;">
                    <span style="color:#FAFAFA; font-weight:600;">{row['driver_name']}</span>
                    <span style="margin-left:8px;">{team_html}</span>
                </div>
                <div style="color:#8B949E; font-size:0.8rem;">Grid: P{int(row['grid'])}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("##### Actual Result")
        actual = race_data.sort_values("positionOrder").head(10)
        for i, (_, row) in enumerate(actual.iterrows()):
            color = get_team_color(row["constructor_name"])
            pos_html = position_badge(i + 1)
            team_html = team_badge(row["constructor_name"], color)
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:8px 12px; margin:4px 0;
                        background:rgba(22,27,34,0.8); border-radius:8px; border-left:3px solid {color};">
                <div style="min-width:50px;">{pos_html}</div>
                <div style="flex:1; margin-left:10px;">
                    <span style="color:#FAFAFA; font-weight:600;">{row['driver_name']}</span>
                    <span style="margin-left:8px;">{team_html}</span>
                </div>
                <div style="color:#8B949E; font-size:0.8rem;">Grid: P{int(row['grid'])}</div>
            </div>
            """, unsafe_allow_html=True)

    # Accuracy check
    st.markdown("<br>", unsafe_allow_html=True)
    pred_winner = pred_df.iloc[0]["driver_name"]
    actual_winner = race_data.sort_values("positionOrder").iloc[0]["driver_name"]
    match = "CORRECT" if pred_winner == actual_winner else "❌"
    st.markdown(f"""
    <div style="background:rgba(22,27,34,0.9); border:1px solid #21262D; border-radius:12px; 
                padding:1rem; text-align:center;">
        <span style="font-size:1.2rem;">{match}</span>
        <span style="color:#C9D1D9; margin-left:10px;">
            Predicted: <strong style="color:#FFD700;">{pred_winner}</strong> | 
            Actual: <strong style="color:#00E676;">{actual_winner}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)


def _render_win_probability(race_data, race_feat, models):
    """Render win probability bar chart."""
    if race_feat.empty:
        st.info("Feature data not available.")
        return

    clf_model = models.get("clf_Random Forest", models.get("clf_XGBoost"))
    scaler_clf = models.get("scaler_clf")
    if not clf_model or not scaler_clf:
        st.warning("Classification model not available.")
        return

    feature_cols_available = [c for c in FEATURE_COLUMNS if c in race_feat.columns]
    X = race_feat[feature_cols_available].copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS].fillna(0)

    X_scaled = pd.DataFrame(scaler_clf.transform(X), columns=X.columns, index=X.index)
    win_prob = clf_model.predict_proba(X_scaled)[:, 1]

    prob_df = race_data[["driver_name", "constructor_name"]].iloc[:len(win_prob)].copy()
    prob_df["win_probability"] = (win_prob * 100).round(2)
    prob_df = prob_df.sort_values("win_probability", ascending=True).tail(10)

    colors = [get_team_color(c) for c in prob_df["constructor_name"]]

    fig = go.Figure(go.Bar(
        x=prob_df["win_probability"],
        y=prob_df["driver_name"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in prob_df["win_probability"]],
        textposition="outside",
        textfont=dict(color="#FAFAFA", size=12, family="Inter"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=0, r=80, t=30, b=0),
        xaxis=dict(title="Win Probability (%)", gridcolor="#21262D", showgrid=True),
        yaxis=dict(showgrid=False),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)
