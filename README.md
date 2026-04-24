# 🏎️ F1 Race Outcome Predictor

> **Can Machine Learning predict an F1 race better than fans?**  
> Using historical data, driver form, constructor pace & track behavior to predict Formula 1 race outcomes.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## The Idea

Train ML models on **historical F1 race data (2009–2026)** to predict:

- **Race Winners** — Who takes the chequered flag?
- **Finishing Positions** — Predicted order for the entire grid
- **Performance Trends** — Driver form & constructor pace across tracks

## What's Behind the Scenes

| Component | Details |
|-----------|---------|
| **Data** | 14 relational CSV files (Ergast-style) covering 2009–2026 |
| **Feature Engineering** | 22 predictive features: driver form, constructor pace, track history, qualifying delta, championship position |
| **Models** | Random Forest, XGBoost, LightGBM (classification + regression) |
| **Evaluation** | Time-based split: Train (2009–2023), Val (2024), Test (2025–2026) |
| **Dashboard** | Premium Streamlit UI with 6 interactive pages |

## Model Performance

### Winner Prediction (Classification)

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest** | 96.70% | 0.6667 | **0.9811** |
| LightGBM | 96.15% | 0.5714 | 0.9748 |
| XGBoost | 95.60% | 0.4783 | 0.9793 |

### Position Prediction (Regression)

| Model | MAE | RMSE |
|-------|-----|------|
| **Random Forest** | **3.41** | **4.34** |
| LightGBM | 3.42 | 4.38 |
| XGBoost | 3.52 | 4.45 |

## 22 Engineered Features

| Category | Features |
|----------|----------|
| Grid & Qualifying | Grid position, qualifying gap to pole, quali position |
| Driver Form | Rolling avg finish (5 races), points momentum, win rate, DNF rate |
| Constructor Pace | Team avg finish, constructor points, team win rate, DNF rate |
| Track History | Circuit avg finish, circuit starts, circuit best result |
| Championship | Driver/constructor championship position, season progress |

## Project Structure

```
f1-race-predictor/
├── data/
│   └── raw/                        # 14 CSV files (Ergast-style dataset)
│       ├── circuits.csv
│       ├── constructors.csv
│       ├── drivers.csv
│       ├── qualifying.csv
│       ├── races.csv
│       ├── results.csv
│       └── ... (8 more)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & merging pipeline
│   ├── feature_engineering.py      # 22 engineered features
│   ├── model_training.py           # Train & evaluate 3 models
│   ├── utils.py                    # Team colors, constants, helpers
│   ├── ui_components.py            # Custom Streamlit UI components
│   └── pages/
│       ├── __init__.py             # Home page
│       ├── predictions.py          # Race predictions page
│       ├── driver_analysis.py      # Driver analysis page
│       ├── constructor_battle.py   # Constructor comparison page
│       ├── model_insights.py       # Model performance page
│       └── season_overview.py      # Season overview page
├── models/                         # Saved trained models (.pkl)
├── .streamlit/
│   └── config.toml                 # Dark theme config
├── app.py                          # Main Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Dhanvin1520/F1-Race-Predictor.git
cd F1-Race-Predictor
pip install -r requirements.txt
```

### 2. Train Models

```bash
python -m src.model_training
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | Project overview, key stats, feature engineering highlights |
| **Race Predictions** | ML-predicted finishing order vs actual results |
| **Driver Analysis** | Driver profiles, performance timeline, track heatmaps |
| **Constructor Battle** | Head-to-head team comparison with points progression |
| **Model Insights** | Model comparison, feature importance, confusion matrices |
| **Season Overview** | Championship standings & race results heatmap |

## Tech Stack

- **Python 3.10+** — Core language
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML preprocessing & Random Forest
- **XGBoost** — Gradient boosted trees
- **LightGBM** — Fast gradient boosting
- **Plotly** — Interactive visualizations
- **Streamlit** — Dashboard framework
- **Joblib** — Model serialization

## Data Source

Historical F1 data sourced from the [Ergast Motor Racing Database](http://ergast.com/mrd/) covering:
- **7,300+** race results from **2009–2026**
- **89** drivers across **28** constructors
- Qualifying, lap times, pit stops, championship standings

## Is F1 Predictable?

After training on 15+ years of data, the answer is: **mostly, yes** — with some chaos.

- Grid position remains the **strongest predictor** of race outcome
- Driver and constructor form provide **significant signal**
- Track-specific history captures **circuit DNA**
- But safety cars, weather, and mechanical failures keep things unpredictable

**ROC-AUC of 0.98** means the model can distinguish winners from non-winners with high confidence.
