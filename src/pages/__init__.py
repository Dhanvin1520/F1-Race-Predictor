"""
pages/home.py — Home Page for F1 Dashboard
"""
import streamlit as st
from src.ui_components import hero_section, stat_card, section_header, tag_badge


def render(df, results):
    """Render the Home page."""
    hero_section()

    # Key stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Races Analyzed", f"{df['raceId'].nunique():,}")
    with c2:
        stat_card("Drivers Tracked", f"{df['driver_name'].nunique()}")
    with c3:
        best_acc = 0
        if results and "classification" in results:
            for m in results["classification"].values():
                best_acc = max(best_acc, m.get("accuracy", 0))
        stat_card("Best Accuracy", f"{best_acc*100:.1f}%")
    with c4:
        best_auc = 0
        if results and "classification" in results:
            for m in results["classification"].values():
                best_auc = max(best_auc, m.get("roc_auc", 0))
        stat_card("Best ROC-AUC", f"{best_auc:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # About section
    col1, col2 = st.columns([2, 1])
    with col1:
        section_header("What This Project Does")
        st.markdown("""
        <div style="background: rgba(22,27,34,0.8); border: 1px solid #21262D; border-radius: 12px; 
                    padding: 1.5rem; margin-top: 0.5rem;">
            <p style="color: #C9D1D9; font-size: 0.95rem; line-height: 1.7;">
                This project uses <strong style="color: #E10600;">Machine Learning</strong> to predict 
                Formula 1 race outcomes. We analyze historical data spanning <strong style="color: #00D2FF;">
                2009–2026</strong> to uncover patterns in:
            </p>
            <ul style="color: #C9D1D9; font-size: 0.9rem; line-height: 2;">
                <li><strong>Race Winners</strong> — Who crosses the line first?</li>
                <li><strong>Finishing Positions</strong> — Predicted order for the entire grid</li>
                <li><strong>Performance Trends</strong> — Driver form & constructor pace across tracks</li>
                <li><strong>Track Behavior</strong> — How drivers perform at specific circuits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section_header("ML Models Used")
        st.markdown("""
        <div style="background: rgba(22,27,34,0.8); border: 1px solid #21262D; border-radius: 12px; 
                    padding: 1.5rem; margin-top: 0.5rem;">
        """, unsafe_allow_html=True)
        
        models_info = [
            ("Random Forest", "#00E676", "Ensemble baseline"),
            ("XGBoost", "#FF8000", "Gradient boosting"),
            ("LightGBM", "#00D2FF", "Fast gradient boosting"),
        ]
        for name, color, desc in models_info:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: {color}; 
                            margin-right: 10px; box-shadow: 0 0 8px {color}80;"></div>
                <div>
                    <div style="color: #FAFAFA; font-weight: 600; font-size: 0.9rem;">{name}</div>
                    <div style="color: #8B949E; font-size: 0.75rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature engineering highlights
    section_header("22 Engineered Features", subtitle="Capturing the DNA of F1 performance")
    
    feat_categories = [
        ("Grid & Qualifying", ["Grid Position", "Qualifying Gap to Pole", "Quali Position"], "#E10600"),
        ("Driver Form", ["Rolling Avg Finish (5 races)", "Points Momentum", "Win Rate", "DNF Rate"], "#00D2FF"),
        ("Constructor Pace", ["Team Avg Finish", "Constructor Points", "Team Win Rate"], "#FF8000"),
        ("Track History", ["Circuit Avg Finish", "Circuit Starts", "Circuit Best"], "#00E676"),
        ("Championship", ["Driver Standing", "Constructor Standing", "Season Progress"], "#FFD700"),
    ]
    
    cols = st.columns(len(feat_categories))
    for col, (cat_name, features, color) in zip(cols, feat_categories):
        with col:
            st.markdown(f"""
            <div style="background: rgba(22,27,34,0.8); border: 1px solid #21262D; border-radius: 12px; 
                        padding: 1rem; height: 100%; border-top: 3px solid {color};">
                <div style="font-weight: 700; color: {color}; font-size: 0.85rem; margin-bottom: 0.8rem;">{cat_name}</div>
            """, unsafe_allow_html=True)
            for f in features:
                st.markdown(f"""
                <div style="color: #C9D1D9; font-size: 0.78rem; padding: 3px 0; 
                            border-bottom: 1px solid #21262D12;">• {f}</div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Tags
    st.markdown("<br>", unsafe_allow_html=True)
    tags = ["#F1", "#MachineLearning", "#DataScience", "#XGBoost", "#LightGBM", 
            "#Python", "#Streamlit", "#Plotly", "#FeatureEngineering", "#Predictions"]
    colors = ["#E10600", "#00D2FF", "#FF8000", "#00E676", "#FFD700", 
              "#E10600", "#00D2FF", "#FF8000", "#00E676", "#FFD700"]
    tag_html = " ".join([tag_badge(t, c) for t, c in zip(tags, colors)])
    st.markdown(f'<div style="text-align: center;">{tag_html}</div>', unsafe_allow_html=True)
