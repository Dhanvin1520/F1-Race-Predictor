"""
utils.py — Helper Functions for F1 Race Outcome Predictor
Provides team color mappings, formatting utilities, and shared constants.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─── Modern Era Start (for model training relevance) ─────────────────────
MODERN_ERA_START = 2009

# ─── F1 Team Color Palette ──────────────────────────────────────────────
TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine F1 Team": "#FF87BC",
    "Williams": "#64C4FF",
    "RB F1 Team": "#6692FF",
    "AlphaTauri": "#4E7C9B",
    "Kick Sauber": "#52E252",
    "Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Alfa Romeo": "#C92D4B",
    "Racing Point": "#F596C8",
    "Renault": "#FFF500",
    "Force India": "#F596C8",
    "Toro Rosso": "#469BFF",
    "Lotus F1": "#FFB800",
    "Marussia": "#6E0000",
    "Caterham": "#005030",
    "Manor Marussia": "#6E0000",
    "BMW Sauber": "#006EFF",
}

# ─── Constructor Name Normalization ─────────────────────────────────────
CONSTRUCTOR_ALIASES = {
    "rb_f1_team": "RB F1 Team",
    "alphatauri": "AlphaTauri",
    "toro_rosso": "Toro Rosso",
    "racing_point": "Racing Point",
    "force_india": "Force India",
    "kick_sauber": "Kick Sauber",
    "alpine": "Alpine F1 Team",
    "alfa": "Alfa Romeo",
    "haas": "Haas F1 Team",
    "red_bull": "Red Bull",
    "mclaren": "McLaren",
    "ferrari": "Ferrari",
    "mercedes": "Mercedes",
    "williams": "Williams",
    "aston_martin": "Aston Martin",
    "renault": "Renault",
    "lotus_f1": "Lotus F1",
    "sauber": "Sauber",
    "bmw_sauber": "BMW Sauber",
    "marussia": "Marussia",
    "caterham": "Caterham",
    "manor_marussia": "Manor Marussia",
}

# ─── Position Badge Colors ──────────────────────────────────────────────
POSITION_COLORS = {
    1: "#FFD700",   # Gold
    2: "#C0C0C0",   # Silver
    3: "#CD7F32",   # Bronze
}

# ─── Dashboard UI Constants ─────────────────────────────────────────────
F1_RED = "#E10600"
F1_DARK = "#0E1117"
F1_CARD = "#161B22"
F1_BLUE = "#00D2FF"
F1_GOLD = "#FFD700"
F1_GREEN = "#00E676"
F1_BORDER = "#30363D"


def get_team_color(constructor_name: str) -> str:
    """Get the hex color for a given constructor/team name."""
    # Direct match
    if constructor_name in TEAM_COLORS:
        return TEAM_COLORS[constructor_name]
    # Check aliases
    for alias_key, alias_val in CONSTRUCTOR_ALIASES.items():
        if alias_key in constructor_name.lower().replace(" ", "_"):
            if alias_val in TEAM_COLORS:
                return TEAM_COLORS[alias_val]
    return "#888888"  # Default gray


def format_time_delta(ms: float) -> str:
    """Format milliseconds to a readable time string."""
    if ms <= 0:
        return "—"
    seconds = ms / 1000
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}:{secs:06.3f}"
    return f"{secs:.3f}s"


def ordinal(n: int) -> str:
    """Return ordinal string for a number (1st, 2nd, 3rd, ...)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
