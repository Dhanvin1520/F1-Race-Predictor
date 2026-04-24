"""
pages/season_overview.py — Season Overview Page
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.ui_components import section_header, stat_card
from src.utils import get_team_color


def render(df):
    """Render the Season Overview page."""
    section_header("Season Overview", subtitle="Championship standings & race-by-race analysis")

    years = sorted(df["year"].unique(), reverse=True)
    sel_year = st.selectbox("Select Season", years, key="season_year")

    year_data = df[df["year"] == sel_year].copy()
    if year_data.empty:
        st.info("No data for this season.")
        return

    # Season stats
    total_races = year_data["raceId"].nunique()
    different_winners = year_data[year_data["positionOrder"] == 1]["driver_name"].nunique()
    total_dnfs = len(year_data[year_data["status"] != "Finished"])

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Races", total_races)
    with c2:
        stat_card("Different Winners", different_winners)
    with c3:
        stat_card("Total DNFs", total_dnfs)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Driver Standings", "Constructor Standings", "Results Heatmap"])

    with tab1:
        _render_driver_standings(year_data, sel_year)
    with tab2:
        _render_constructor_standings(year_data, sel_year)
    with tab3:
        _render_heatmap(year_data, sel_year)


def _render_driver_standings(data, year):
    """Driver championship standings."""
    standings = data.groupby(["driverId", "driver_name", "constructor_name"]).agg(
        total_points=("points", "sum"),
        wins=("is_winner", "sum") if "is_winner" in data.columns else ("positionOrder", lambda x: (x == 1).sum()),
        races=("raceId", "nunique"),
    ).reset_index()
    standings = standings.sort_values("total_points", ascending=False).reset_index(drop=True)
    standings.index = standings.index + 1

    colors = [get_team_color(c) for c in standings["constructor_name"]]

    fig = go.Figure(go.Bar(
        x=standings["total_points"].head(15),
        y=standings["driver_name"].head(15),
        orientation="h",
        marker=dict(color=colors[:15]),
        text=[f"{p:.0f} pts" for p in standings["total_points"].head(15)],
        textposition="outside",
        textfont=dict(color="#FAFAFA", size=11, family="Inter"),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500, yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(title="Points", gridcolor="#21262D"),
        font=dict(family="Inter"), margin=dict(l=0, r=80, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_constructor_standings(data, year):
    """Constructor championship standings."""
    standings = data.groupby("constructor_name").agg(
        total_points=("points", "sum"),
        races=("raceId", "nunique"),
    ).reset_index()
    standings = standings.sort_values("total_points", ascending=False)

    colors = [get_team_color(c) for c in standings["constructor_name"]]

    fig = go.Figure(go.Bar(
        x=standings["total_points"],
        y=standings["constructor_name"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{p:.0f} pts" for p in standings["total_points"]],
        textposition="outside",
        textfont=dict(color="#FAFAFA", size=12, family="Inter"),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(title="Points", gridcolor="#21262D"),
        font=dict(family="Inter"), margin=dict(l=0, r=80, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(data, year):
    """Race results heatmap for top drivers."""
    # Get top 10 drivers by points
    top_drivers = data.groupby("driver_name")["points"].sum().nlargest(10).index.tolist()
    races = data[["raceId", "race_name", "round"]].drop_duplicates().sort_values("round")

    heatmap_data = []
    for driver in top_drivers:
        row = {"driver": driver}
        for _, race in races.iterrows():
            result = data[(data["driver_name"] == driver) & (data["raceId"] == race["raceId"])]
            pos = result["positionOrder"].values[0] if not result.empty else None
            row[race["race_name"]] = pos
        heatmap_data.append(row)

    hm_df = pd.DataFrame(heatmap_data).set_index("driver")

    fig = go.Figure(go.Heatmap(
        z=hm_df.values,
        x=[n[:12] for n in hm_df.columns],
        y=hm_df.index,
        colorscale=[[0, "#FFD700"], [0.15, "#00E676"], [0.5, "#00D2FF"], [0.8, "#FF8000"], [1, "#E10600"]],
        text=hm_df.values,
        texttemplate="P%{text}",
        textfont=dict(size=10, color="#FAFAFA"),
        zmin=1, zmax=20,
        colorbar=dict(title="Position", tickvals=[1, 5, 10, 15, 20]),
        hoverongaps=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450, xaxis=dict(tickangle=45, showgrid=False),
        yaxis=dict(showgrid=False), font=dict(family="Inter"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
