"""Prediction results with quantile bands page."""

import plotly.graph_objects as go
import streamlit as st

from utils.constants import CONFIG_IDS, MODEL_COLORS, PLOTLY_LAYOUT
from utils.data_loader import get_scatter_image, load_dl_results

st.title("예측 결과")
st.caption("DL 모델의 분위수 예측 밴드 (p10/p50/p90) 및 실제 vs 예측 비교")

selected = st.sidebar.selectbox("품목 선택", CONFIG_IDS, key="pred_config")

# ── Load data ─────────────────────────────────────────────────────────
dl = load_dl_results()
quantile_results = dl.get("quantile_results", {})
config_quantile = quantile_results.get(selected, {})

# ── Quantile Metrics Table ────────────────────────────────────────────
if config_quantile:
    st.subheader(f"{selected} — 분위수 예측 성능")

    import pandas as pd

    rows = []
    for model_name, metrics in config_quantile.items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            "모델": model_name,
            "MAPE (p50)": f"{metrics.get('mape_p50', '-')}%",
            "밴드 커버리지": f"{metrics.get('coverage', '-')}%",
            "밴드 폭 (원)": f"{metrics.get('band_width_avg', 0):,.0f}",
            "밴드 비율": f"{metrics.get('band_pct', '-')}%",
            "CQR 커버리지": f"{metrics.get('conformal_coverage', '-')}%",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    # ── Example Forecast Visualization ────────────────────────────────
    st.subheader("예측 밴드 예시")
    st.caption("각 모델의 마지막 예측 시점 p10/p50/p90 밴드와 실제 가격 비교")

    models_with_forecast = [
        (m, d) for m, d in config_quantile.items()
        if isinstance(d, dict) and "example_forecast" in d
    ]

    if models_with_forecast:
        fig = go.Figure()

        for i, (model_name, metrics) in enumerate(models_with_forecast):
            fc = metrics["example_forecast"]
            p10, p50, p90, actual = fc.get("p10", 0), fc.get("p50", 0), fc.get("p90", 0), fc.get("actual", 0)
            color = MODEL_COLORS.get(model_name, "#999")

            # Band (p10 to p90)
            fig.add_trace(go.Bar(
                x=[model_name], y=[p90 - p10],
                base=[p10],
                name=f"{model_name} 밴드",
                marker_color=color, opacity=0.3,
                showlegend=False,
                width=0.4,
            ))
            # p50 marker
            fig.add_trace(go.Scatter(
                x=[model_name], y=[p50],
                mode="markers", name=f"{model_name} p50",
                marker=dict(color=color, size=12, symbol="diamond"),
                showlegend=False,
            ))
            # Actual marker
            fig.add_trace(go.Scatter(
                x=[model_name], y=[actual],
                mode="markers", name="실제",
                marker=dict(color="black", size=10, symbol="x"),
                showlegend=False if i > 0 else True,
            ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{selected} — 모델별 예측 밴드 vs 실제",
            yaxis_title="가격 (원)",
            height=450,
            barmode="overlay",
        )
        # Add legend for actual
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="black", size=10, symbol="x"),
            name="실제 가격",
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="gray", size=12, symbol="diamond"),
            name="p50 예측",
        ))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info(f"{selected}의 분위수 예측 데이터가 없습니다.")

st.markdown("---")

# ── Static Time Series & Scatter Plots ────────────────────────────────
st.subheader("시계열 예측 차트")
st.caption("실제 가격 (검정), 예측 가격 (파랑), p10-p90 밴드 (음영)")

col1, col2 = st.columns(2)

ts_img = get_scatter_image(selected, "timeseries")
sc_img = get_scatter_image(selected, "scatter")

with col1:
    if ts_img:
        st.image(str(ts_img), caption=f"{selected} — 시계열", use_container_width=True)
    else:
        st.info("시계열 플롯이 없습니다.")

with col2:
    if sc_img:
        st.image(str(sc_img), caption=f"{selected} — 실제 vs 예측", use_container_width=True)
    else:
        st.info("산점도 플롯이 없습니다.")
