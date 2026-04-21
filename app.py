"""
🏎️ F1 Race Outcome Predictor — Streamlit Dashboard
Main entry point for the premium F1 prediction dashboard.
"""

import streamlit as st
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.model_training import load_trained_models, load_results
from src.ui_components import inject_css, sidebar_nav

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor | ML-Powered",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject Custom CSS ──────────────────────────────────────────────────
inject_css()


# ─── Cache Data Loading ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading F1 data...")
def get_data():
    df = load_data()
    return df


@st.cache_data(ttl=3600, show_spinner="Engineering features...")
def get_featured_data(_df):
    return engineer_features(_df)


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return load_trained_models()


@st.cache_data(ttl=3600)
def get_results():
    return load_results()


# ─── Load Everything ────────────────────────────────────────────────────
df = get_data()
df_featured = get_featured_data(df)
models = get_models()
results = get_results()

# ─── Sidebar Navigation ─────────────────────────────────────────────────
page = sidebar_nav()

# ─── Page Routing ────────────────────────────────────────────────────────
if page == "Home":
    from src.pages import render as render_home
    render_home(df, results)

elif page == "Race Predictions":
    from src.pages.predictions import render as render_predictions
    render_predictions(df, df_featured, models, results)

elif page == "Driver Analysis":
    from src.pages.driver_analysis import render as render_driver
    render_driver(df)

elif page == "Constructor Battle":
    from src.pages.constructor_battle import render as render_constructor
    render_constructor(df)

elif page == "Model Insights":
    from src.pages.model_insights import render as render_insights
    render_insights(results)

elif page == "Season Overview":
    from src.pages.season_overview import render as render_season
    render_season(df)
