"""
pages/model_insights.py — Model Performance & Insights Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.ui_components import section_header, stat_card


def render(results):
    """Render the Model Insights page."""
    section_header("Model Insights", subtitle="Performance metrics, feature importance & evaluation")

    if not results:
        st.warning("No training results found. Train models first.")
        return

    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Feature Importance", "Evaluation Details"])

    with tab1:
        _render_comparison(results)
    with tab2:
        _render_feature_importance(results)
    with tab3:
        _render_evaluation(results)


def _render_comparison(results):
    """Model performance comparison table and charts."""
    # Classification results
    st.markdown("##### 🏆 Winner Prediction (Classification)")
    clf_results = results.get("classification", {})
    if clf_results:
        rows = []
        for name, metrics in clf_results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                "F1 Score": f"{metrics['f1_score']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            })
        st.markdown(_styled_table(pd.DataFrame(rows)), unsafe_allow_html=True)

        # Bar chart comparison
        models = list(clf_results.keys())
        metrics_names = ["accuracy", "f1_score", "roc_auc"]
        colors = ["#E10600", "#00D2FF", "#FFD700"]

        fig = go.Figure()
        for metric, color in zip(metrics_names, colors):
            values = [clf_results[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(
                name=metric.replace("_", " ").title(),
                x=models, y=values,
                marker_color=color, text=[f"{v:.3f}" for v in values],
                textposition="outside", textfont=dict(color="#FAFAFA", size=11),
            ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, barmode="group", font=dict(family="Inter"),
            yaxis=dict(gridcolor="#21262D", range=[0, 1.15]),
            legend=dict(orientation="h", y=1.15), margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Regression results
    st.markdown("##### 📍 Position Prediction (Regression)")
    reg_results = results.get("regression", {})
    if reg_results:
        rows = []
        for name, metrics in reg_results.items():
            rows.append({
                "Model": name,
                "MAE": f"{metrics['mae']:.4f}",
                "RMSE": f"{metrics['rmse']:.4f}",
            })
        st.markdown(_styled_table(pd.DataFrame(rows)), unsafe_allow_html=True)

    # Best models callout
    best_clf = results.get("best_classifier", "N/A")
    best_reg = results.get("best_regressor", "N/A")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(225,6,0,0.1), rgba(255,215,0,0.1));
                border:1px solid #21262D; border-radius:12px; padding:1.2rem; margin-top:1rem;
                text-align:center;">
        <span style="font-size:1.1rem; color:#FAFAFA; font-weight:600;">
            Best Classifier: <span style="color:#FFD700;">{best_clf}</span> &nbsp;|&nbsp;
            Best Regressor: <span style="color:#FFD700;">{best_reg}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def _render_feature_importance(results):
    """Feature importance bar charts."""
    fi = results.get("feature_importances", {})
    if not fi:
        st.info("No feature importance data available.")
        return

    # Pick classification models
    clf_keys = [k for k in fi.keys() if k.startswith("clf_")]
    if not clf_keys:
        clf_keys = list(fi.keys())

    for key in clf_keys:
        model_name = key.replace("clf_", "").replace("reg_", "")
        importance = fi[key]
        fi_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#161B22"], [0.5, "#E10600"], [1, "#FFD700"]],
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside", textfont=dict(color="#C9D1D9", size=10),
        ))
        fig.update_layout(
            title=dict(text=f"{model_name} — Top Features", font=dict(size=14, color="#FAFAFA")),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=450, xaxis=dict(gridcolor="#21262D", title="Importance"),
            yaxis=dict(showgrid=False), font=dict(family="Inter"),
            margin=dict(l=0, r=80, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_evaluation(results):
    """Confusion matrix and data info."""
    clf_results = results.get("classification", {})

    # Confusion matrices
    for name, metrics in clf_results.items():
        cm = metrics.get("confusion_matrix")
        if cm:
            cm = np.array(cm)
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Not Winner", "Winner"], y=["Not Winner", "Winner"],
                colorscale=[[0, "#161B22"], [1, "#E10600"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=16, color="#FAFAFA"),
                showscale=False,
            ))
            fig.update_layout(
                title=dict(text=f"{name} — Confusion Matrix", font=dict(size=14, color="#FAFAFA")),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, xaxis=dict(title="Predicted"), yaxis=dict(title="Actual", autorange="reversed"),
                font=dict(family="Inter"), margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Data info
    data_info = results.get("data_info", {})
    if data_info:
        st.markdown("##### 📋 Training Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stat_card("Training Period", data_info.get("train_years", "N/A"))
        with col2:
            stat_card("Test Period", data_info.get("test_years", "N/A"))
        with col3:
            stat_card("Total Samples", f"{data_info.get('total_samples', 0):,}")
        with col4:
            stat_card("Features", data_info.get("n_features", 0))


def _styled_table(df):
    """Return styled HTML table."""
    header = "".join([f'<th style="padding:10px 15px; text-align:left; color:#8B949E; font-size:0.8rem; '
                      f'text-transform:uppercase; letter-spacing:0.05em; border-bottom:2px solid #21262D;">{c}</th>'
                      for c in df.columns])
    rows_html = ""
    for _, row in df.iterrows():
        cells = "".join([f'<td style="padding:10px 15px; color:#C9D1D9; font-weight:500; '
                         f'border-bottom:1px solid #21262D15;">{v}</td>' for v in row])
        rows_html += f'<tr style="transition:all 0.2s;">{cells}</tr>'

    return f"""
    <div style="background:rgba(22,27,34,0.8); border:1px solid #21262D; border-radius:12px; overflow:hidden; margin:0.5rem 0;">
        <table style="width:100%; border-collapse:collapse;">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
