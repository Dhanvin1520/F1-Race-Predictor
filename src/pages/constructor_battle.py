"""
pages/constructor_battle.py — Constructor Battle Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui_components import section_header, stat_card
from src.utils import get_team_color


def render(df):
    """Render the Constructor Battle page."""
    section_header("Constructor Battle", subtitle="Head-to-head team performance comparison")

    # Year selector
    years = sorted(df["year"].unique(), reverse=True)
    sel_year = st.selectbox("Select Season", years, key="con_year")

    year_data = df[df["year"] == sel_year].copy()
    teams = sorted(year_data["constructor_name"].unique())

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0, key="team1")
    with col2:
        team2 = st.selectbox("Team 2", teams, index=min(1, len(teams) - 1), key="team2")

    st.markdown("---")

    t1_data = year_data[year_data["constructor_name"] == team1]
    t2_data = year_data[year_data["constructor_name"] == team2]

    # Comparison stats
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics = [
        ("Total Points", t1_data["points"].sum(), t2_data["points"].sum(), ""),
        ("Wins", len(t1_data[t1_data["positionOrder"] == 1]), len(t2_data[t2_data["positionOrder"] == 1]), ""),
        ("Podiums", len(t1_data[t1_data["positionOrder"] <= 3]), len(t2_data[t2_data["positionOrder"] <= 3]),""),
        ("Avg Finish", f"{t1_data['positionOrder'].mean():.1f}", f"{t2_data['positionOrder'].mean():.1f}", "📍"),
        ("Best Finish", f"P{int(t1_data['positionOrder'].min())}" if not t1_data.empty else "—",
         f"P{int(t2_data['positionOrder'].min())}" if not t2_data.empty else "—", ""),
        ("DNFs", len(t1_data[t1_data["status"] != "Finished"]),
         len(t2_data[t2_data["status"] != "Finished"]),""),
    ]

    for col, (label, v1, v2, icon) in zip([col1, col2, col3, col4, col5, col6], metrics):
        with col:
            c1 = get_team_color(team1)
            c2 = get_team_color(team2)
            st.markdown(f"""
            <div style="background:rgba(22,27,34,0.9); border:1px solid #21262D; border-radius:12px; padding:1rem;">
                <div style="color:#8B949E; font-size:0.7rem; font-weight:600; text-transform:uppercase; margin-bottom:0.5rem;">
                    {icon} {label}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:{c1}; font-weight:800; font-size:1.2rem;">{v1}</span>
                    <span style="color:#30363D;">vs</span>
                    <span style="color:{c2}; font-weight:800; font-size:1.2rem;">{v2}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Points Progression", "Race-by-Race"])

    with tab1:
        _render_points_progression(year_data, team1, team2, sel_year)

    with tab2:
        _render_race_comparison(year_data, team1, team2)


def _render_points_progression(data, team1, team2, year):
    """Cumulative points over the season."""
    fig = go.Figure()

    for team in [team1, team2]:
        team_data = data[data["constructor_name"] == team].sort_values("date")
        race_points = team_data.groupby(["raceId", "race_name", "round"])["points"].sum().reset_index()
        race_points = race_points.sort_values("round")
        race_points["cumulative"] = race_points["points"].cumsum()
        color = get_team_color(team)

        fig.add_trace(go.Scatter(
            x=race_points["race_name"], y=race_points["cumulative"],
            mode="lines+markers",
            name=team,
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color, line=dict(width=2, color="#0E1117")),
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        yaxis=dict(title="Cumulative Points", gridcolor="#21262D"),
        xaxis=dict(tickangle=45, showgrid=False),
        font=dict(family="Inter"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_race_comparison(data, team1, team2):
    """Best finish per race comparison."""
    rounds = data[["raceId", "race_name", "round"]].drop_duplicates().sort_values("round")

    results = []
    for _, race in rounds.iterrows():
        rid = race["raceId"]
        rname = race["race_name"]
        t1 = data[(data["raceId"] == rid) & (data["constructor_name"] == team1)]
        t2 = data[(data["raceId"] == rid) & (data["constructor_name"] == team2)]
        t1_best = t1["positionOrder"].min() if not t1.empty else None
        t2_best = t2["positionOrder"].min() if not t2.empty else None
        results.append({"race": rname, team1: t1_best, team2: t2_best})

    res_df = pd.DataFrame(results)

    fig = go.Figure()
    c1 = get_team_color(team1)
    c2 = get_team_color(team2)

    fig.add_trace(go.Scatter(
        x=res_df["race"], y=res_df[team1], mode="lines+markers",
        name=team1, line=dict(color=c1, width=2), marker=dict(size=8, color=c1),
    ))
    fig.add_trace(go.Scatter(
        x=res_df["race"], y=res_df[team2], mode="lines+markers",
        name=team2, line=dict(color=c2, width=2), marker=dict(size=8, color=c2),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, yaxis=dict(autorange="reversed", title="Best Finish Position", gridcolor="#21262D"),
        xaxis=dict(tickangle=45, showgrid=False), font=dict(family="Inter"),
        legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
