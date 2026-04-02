"""Interactive price time series page — all species, full period."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.charts import seasonality_heatmap, timeseries_chart
from utils.constants import PLOTLY_LAYOUT
from utils.data_loader import get_all_species, load_all_species_daily

st.title("시세 현황")
st.caption("전체 어종의 일별 경락 가격 추이, 이동평균, 거래량, 계절성 분석")

# ── Load data ─────────────────────────────────────────────────────────
all_species = get_all_species()
daily = load_all_species_daily()

if daily.empty:
    st.error("Parquet 데이터를 불러올 수 없습니다. data/parquet/prices/ 디렉토리를 확인해 주세요.")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────
selected = st.sidebar.selectbox(
    "어종 선택",
    all_species,
    index=all_species.index("넙치") if "넙치" in all_species else 0,
    key="price_species",
)

sub = daily[daily["species"] == selected].copy()
if sub.empty:
    st.warning(f"{selected}에 대한 데이터가 없습니다.")
    st.stop()

sub = sub.sort_values("trade_date_dt")
min_date = sub["trade_date_dt"].min().date()
max_date = sub["trade_date_dt"].max().date()

# Default: full period
date_range = st.sidebar.date_input(
    "기간 선택",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = date_range
else:
    start_dt, end_dt = min_date, max_date

price_col = st.sidebar.radio(
    "가격 유형",
    ["price_avg", "price_high", "price_low"],
    format_func=lambda x: {"price_avg": "평균가", "price_high": "고가", "price_low": "저가"}[x],
)
show_ma = st.sidebar.checkbox("이동평균 표시", value=True)

# Filter by date range
mask = (sub["trade_date_dt"].dt.date >= start_dt) & (sub["trade_date_dt"].dt.date <= end_dt)
filtered = sub[mask].copy()

if filtered.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

# ── Time Series Chart ─────────────────────────────────────────────────
fig = timeseries_chart(
    filtered,
    price_col=price_col,
    title=f"{selected} 일별 가격 추이 ({start_dt} ~ {end_dt})",
    show_ma=show_ma,
)
st.plotly_chart(fig, use_container_width=True)

# ── Summary Stats ─────────────────────────────────────────────────────
st.subheader("기간 통계 요약")

prices = filtered[price_col].dropna()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("평균가", f"{prices.mean():,.0f}원")
col2.metric("표준편차", f"{prices.std():,.0f}원")
col3.metric("변동계수 (CV)", f"{prices.std() / prices.mean():.3f}" if prices.mean() > 0 else "N/A")

if len(prices) > 2:
    lag1 = prices.autocorr(lag=1)
    col4.metric("Lag-1 자기상관", f"{lag1:.3f}" if not np.isnan(lag1) else "N/A")
else:
    col4.metric("Lag-1 자기상관", "N/A")

col5.metric("거래일수", f"{len(filtered):,}일")

st.markdown("---")

# ── Seasonality Heatmap ───────────────────────────────────────────────
st.subheader("월별 계절성 히트맵")
st.caption("연도 x 월 평균가격. 색이 진할수록 가격이 높은 시기입니다.")

fig_heat = seasonality_heatmap(sub, price_col=price_col, title=f"{selected} — 월별 평균 가격")
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── Price Distribution ────────────────────────────────────────────────
st.subheader("가격 분포")

fig_hist = go.Figure(go.Histogram(
    x=filtered[price_col],
    nbinsx=50,
    marker_color="#1f77b4",
    opacity=0.7,
))
fig_hist.update_layout(
    **PLOTLY_LAYOUT,
    title=f"{selected} — 가격 분포 ({start_dt} ~ {end_dt})",
    xaxis_title="가격 (원)",
    yaxis_title="빈도",
    height=350,
)
st.plotly_chart(fig_hist, use_container_width=True)
