"""
ui_components.py — Reusable UI Components for the F1 Dashboard
Custom HTML/CSS components for the premium Streamlit interface.
"""

import streamlit as st

# ─── Custom CSS for the entire dashboard ─────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #0A0D12 100%);
}

/* Hide default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #12161D 100%) !important;
    border-right: 1px solid #1E2530 !important;
}
[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600 !important;
    color: #FAFAFA !important;
}

/* Metric cards */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #E10600 0%, #FF4136 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #8B949E !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Containers / Cards */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #21262D !important;
    border-radius: 12px !important;
    background: rgba(22, 27, 34, 0.8) !important;
    backdrop-filter: blur(10px) !important;
}

/* Selectbox / Input styling */
.stSelectbox > div > div {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    background-color: #161B22 !important;
    border: 1px solid #21262D !important;
    color: #8B949E !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E10600 0%, #FF4136 100%) !important;
    border-color: #E10600 !important;
    color: #FFFFFF !important;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0E1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #E10600; }

/* Divider */
hr { border-color: #21262D !important; }

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #E10600 0%, #FF4136 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 2rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(225, 6, 0, 0.3) !important;
}
</style>
"""


def inject_css():
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero_section():
    """Render the hero section with gradient text and stats."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <div style="font-size: 3.5rem; font-weight: 900; line-height: 1.1; margin-bottom: 0.5rem;
                    background: linear-gradient(135deg, #E10600 0%, #FF4136 30%, #FFD700 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏎️ F1 Race Predictor
        </div>
        <div style="font-size: 1.1rem; color: #8B949E; font-weight: 400; max-width: 600px; margin: 0.5rem auto;">
            Machine Learning meets Formula 1 — Predicting race outcomes using historical data, 
            driver form, constructor pace & track behavior
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label, value, delta=None, icon=""):
    """Render a custom stat card with glassmorphism."""
    delta_html = ""
    if delta is not None:
        color = "#00E676" if delta >= 0 else "#FF4136"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem; font-weight: 600;">{arrow} {abs(delta):.1f}%</div>'
    
    st.markdown(f"""
    <div style="background: rgba(22, 27, 34, 0.9); border: 1px solid #21262D; border-radius: 12px;
                padding: 1.2rem; backdrop-filter: blur(10px); transition: all 0.3s ease;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
        <div style="color: #8B949E; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.3rem;">{icon} {label}</div>
        <div style="font-size: 1.8rem; font-weight: 800;
                    background: linear-gradient(135deg, #FFFFFF 0%, #C9D1D9 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle="", icon=""):
    """Render a styled section header."""
    sub_html = f'<div style="color: #8B949E; font-size: 0.9rem; margin-top: 0.2rem;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #FAFAFA;">
            {icon} {title}
        </div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def position_badge(pos):
    """Return HTML for a position badge (P1 gold, P2 silver, P3 bronze)."""
    colors = {1: ("#FFD700", "#000"), 2: ("#C0C0C0", "#000"), 3: ("#CD7F32", "#FFF")}
    bg, fg = colors.get(pos, ("#30363D", "#FFF"))
    return f"""<span style="display:inline-block; background:{bg}; color:{fg}; 
               border-radius:6px; padding:2px 10px; font-weight:700; font-size:0.85rem;
               min-width:35px; text-align:center;">P{pos}</span>"""


def team_badge(team_name, color):
    """Return HTML for a team color badge."""
    return f"""<span style="display:inline-block; background:{color}20; color:{color}; 
               border:1px solid {color}60; border-radius:20px; padding:3px 12px; 
               font-weight:600; font-size:0.75rem;">{team_name}</span>"""


def tag_badge(text, color="#E10600"):
    """Return HTML for a generic tag badge."""
    return f"""<span style="display:inline-block; background:{color}15; color:{color}; 
               border:1px solid {color}40; border-radius:20px; padding:3px 12px; 
               font-weight:600; font-size:0.75rem; margin:2px;">{text}</span>"""


def sidebar_nav():
    """Render sidebar navigation and return selected page."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 1.8rem; font-weight: 900;
                        background: linear-gradient(135deg, #E10600, #FFD700);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🏎️ F1 Predictor
            </div>
            <div style="color: #8B949E; font-size: 0.75rem; font-weight: 500; margin-top: 0.2rem;">
                ML-Powered Race Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigate",
            ["Home", "Race Predictions", "Driver Analysis",
             "Constructor Battle", "Model Insights", "Season Overview"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Tags section
        st.markdown("""
        <div style="padding: 0.5rem 0;">
            <div style="color: #8B949E; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
                        letter-spacing: 0.1em; margin-bottom: 0.5rem;">TECH STACK</div>
        </div>
        """, unsafe_allow_html=True)
        
        tags = ["Python", "XGBoost", "LightGBM", "Scikit-Learn", "Plotly", "Streamlit"]
        tag_html = " ".join([
            f'<span style="display:inline-block; background:#E1060015; color:#E10600; '
            f'border:1px solid #E1060040; border-radius:20px; padding:2px 10px; '
            f'font-weight:600; font-size:0.65rem; margin:2px;">{t}</span>'
            for t in tags
        ])
        st.markdown(tag_html, unsafe_allow_html=True)
        

    
    return page
