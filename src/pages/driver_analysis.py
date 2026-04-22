"""
pages/driver_analysis.py — Driver Analysis Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui_components import section_header, stat_card, position_badge, team_badge
from src.utils import get_team_color


def render(df):
    """Render the Driver Analysis page."""
    section_header("Driver Analysis", subtitle="Deep-dive into driver performance & form")

    # Driver selector
    recent_drivers = df[df["year"] >= 2020]["driver_name"].unique()
    all_drivers = sorted(df["driver_name"].unique())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        driver = st.selectbox("Select Driver", sorted(recent_drivers), key="driver_sel")
    
    driver_data = df[df["driver_name"] == driver].copy()
    if driver_data.empty:
        st.info("No data for selected driver.")
        return

    latest = driver_data.sort_values("date").iloc[-1]

    # Profile card
    st.markdown(f"""
    <div style="background:rgba(22,27,34,0.9); border:1px solid #21262D; border-radius:16px; 
                padding:1.5rem; margin:1rem 0; display:flex; align-items:center; gap:1.5rem;
                border-left:4px solid {get_team_color(latest['constructor_name'])};">
        <div style="flex:1;">
            <div style="font-size:1.8rem; font-weight:800; color:#FAFAFA;">{driver}</div>
            <div style="color:{get_team_color(latest['constructor_name'])}; font-weight:600; font-size:1rem; margin-top:0.2rem;">
                {latest['constructor_name']}</div>
            <div style="color:#8B949E; font-size:0.85rem; margin-top:0.3rem;">
                {latest.get('driver_nationality', 'N/A')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    total_races = len(driver_data)
    wins = len(driver_data[driver_data["positionOrder"] == 1])
    podiums = len(driver_data[driver_data["positionOrder"] <= 3])
    dnfs = len(driver_data[driver_data["status"] != "Finished"])
    avg_finish = driver_data["positionOrder"].mean()
    total_points = driver_data["points"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        stat_card("Races", total_races)
    with c2:
        stat_card("Wins", wins)
    with c3:
        stat_card("Podiums", podiums)
    with c4:
        stat_card("DNFs", dnfs)
    with c5:
        stat_card("Avg Finish", f"{avg_finish:.1f}")
    with c6:
        stat_card("Total Points", f"{total_points:.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Performance Timeline", "Track Heatmap", "Form Analysis"])

    with tab1:
        _render_timeline(driver_data, driver)

    with tab2:
        _render_track_heatmap(driver_data, driver)

    with tab3:
        _render_form(driver_data, driver)


def _render_timeline(data, driver):
    """Finishing position over time."""
    timeline = data[["date", "race_name", "positionOrder", "grid", "constructor_name", "year"]].copy()
    timeline = timeline.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["date"], y=timeline["positionOrder"],
        mode="lines+markers",
        line=dict(color="#E10600", width=2),
        marker=dict(size=6, color="#E10600", line=dict(width=1, color="#FAFAFA")),
        name="Finish Position",
        hovertemplate="<b>%{customdata[0]}</b><br>Position: P%{y}<br>Grid: P%{customdata[1]}<extra></extra>",
        customdata=timeline[["race_name", "grid"]].values,
    ))
    fig.add_trace(go.Scatter(
        x=timeline["date"], y=timeline["grid"],
        mode="lines",
        line=dict(color="#00D2FF", width=1, dash="dot"),
        name="Grid Position",
        opacity=0.5,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, yaxis=dict(autorange="reversed", title="Position", gridcolor="#21262D"),
        xaxis=dict(gridcolor="#21262D", title=""), font=dict(family="Inter"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_track_heatmap(data, driver):
    """Average finish by circuit."""
    track_perf = data.groupby("circuit_name").agg(
        avg_pos=("positionOrder", "mean"),
        races=("raceId", "count"),
        best=("positionOrder", "min"),
    ).reset_index()
    track_perf = track_perf[track_perf["races"] >= 2].sort_values("avg_pos")

    colors = ["#00E676" if v <= 5 else "#FFD700" if v <= 10 else "#FF4136" for v in track_perf["avg_pos"]]

    fig = go.Figure(go.Bar(
        x=track_perf["avg_pos"],
        y=track_perf["circuit_name"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"P{v:.1f} ({r} races)" for v, r in zip(track_perf["avg_pos"], track_perf["races"])],
        textposition="outside",
        textfont=dict(color="#C9D1D9", size=11, family="Inter"),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(track_perf) * 30),
        xaxis=dict(title="Average Finishing Position", gridcolor="#21262D"),
        yaxis=dict(showgrid=False), font=dict(family="Inter"),
        margin=dict(l=0, r=100, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_form(data, driver):
    """Rolling form analysis."""
    form = data.sort_values("date").copy()
    form["rolling_avg"] = form["positionOrder"].rolling(5, min_periods=1).mean()
    form["rolling_points"] = form["points"].rolling(5, min_periods=1).sum()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=form["date"], y=form["rolling_avg"],
            fill="tozeroy", fillcolor="rgba(225,6,0,0.1)",
            line=dict(color="#E10600", width=2),
            name="5-Race Avg Position",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, yaxis=dict(autorange="reversed", title="Avg Position", gridcolor="#21262D"),
            xaxis=dict(gridcolor="#21262D"), font=dict(family="Inter"),
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=form["date"], y=form["rolling_points"],
            marker=dict(color="#00D2FF", opacity=0.7),
            name="5-Race Points",
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, yaxis=dict(title="Rolling Points", gridcolor="#21262D"),
            xaxis=dict(gridcolor="#21262D"), font=dict(family="Inter"),
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
