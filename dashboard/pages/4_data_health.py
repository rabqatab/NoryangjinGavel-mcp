"""Data health monitoring page."""

import pandas as pd
import streamlit as st

from utils.charts import mape_bar_chart, quality_scatter
from utils.constants import BEST_OF_BREED, PLOTLY_LAYOUT
from utils.data_loader import load_config_registry

st.title("데이터 건강 현황")
st.caption("20개 모니터링 품목의 데이터 품질, 신선도, 거래량 현황")

registry = load_config_registry()
if not registry:
    st.error("prediction_config_registry.json을 찾을 수 없습니다.")
    st.stop()

# ── Freshness badges ──────────────────────────────────────────────────
st.subheader("데이터 신선도")

fresh_cols = st.columns(3)
active = [c for c in registry if c.get("recent_days_2025", 0) >= 100]
caution = [c for c in registry if 1 <= c.get("recent_days_2025", 0) < 100]
inactive = [c for c in registry if c.get("recent_days_2025", 0) == 0]

with fresh_cols[0]:
    st.metric("최신", f"{len(active)}개", help="2025년 거래일 100일 이상")
    for c in active:
        st.markdown(f"- {c['id']} ({c['recent_days_2025']}일)")

with fresh_cols[1]:
    st.metric("주의", f"{len(caution)}개", help="2025년 거래일 1~99일")
    for c in caution:
        st.markdown(f"- {c['id']} ({c['recent_days_2025']}일)")

with fresh_cols[2]:
    st.metric("중단", f"{len(inactive)}개", help="2025년 거래 데이터 없음")
    for c in inactive:
        date_range = c.get("date_range", "")
        last = date_range.split("~")[-1].strip() if "~" in date_range else "N/A"
        st.markdown(f"- {c['id']} (마지막: {last})")

st.markdown("---")

# ── Health table ──────────────────────────────────────────────────────
st.subheader("품목별 상세 현황")

rows = []
for c in registry:
    recent = c.get("recent_days_2025", 0)
    if recent >= 100:
        status = "최신"
    elif recent >= 1:
        status = "주의"
    else:
        status = "중단"

    mape_info = BEST_OF_BREED.get(c["id"], {})

    rows.append({
        "품목 ID": c["id"],
        "어종": c.get("species", ""),
        "상태": c.get("state", ""),
        "규격": c.get("spec", ""),
        "총 건수": c.get("total_rows", 0),
        "거래일수": c.get("trading_days", 0),
        "평균가": f"{c.get('mean_price', 0):,}원",
        "CV": round(c.get("cv", 0), 3),
        "Lag-1": round(c.get("lag1", 0), 3),
        "일평균 건수": round(c.get("mean_lots_per_day", 0), 1),
        "2025 거래일": recent,
        "신선도": status,
        "최고 MAPE": f"{mape_info.get('mape', '-')}%",
        "최고 모델": mape_info.get("model", "-"),
    })

df = pd.DataFrame(rows)

def highlight_freshness(val):
    if val == "최신":
        return "background-color: #d4edda"
    elif val == "주의":
        return "background-color: #fff3cd"
    else:
        return "background-color: #f8d7da"

styled = df.style.map(highlight_freshness, subset=["신선도"])
st.dataframe(styled, use_container_width=True, height=600)

st.markdown("---")

# ── Volume chart ──────────────────────────────────────────────────────
st.subheader("품목별 총 데이터 건수")
import plotly.graph_objects as go

sorted_reg = sorted(registry, key=lambda c: c.get("total_rows", 0), reverse=True)
fig_vol = go.Figure(go.Bar(
    y=[c["id"] for c in sorted_reg],
    x=[c.get("total_rows", 0) for c in sorted_reg],
    orientation="h",
    marker_color="#1f77b4",
    text=[f"{c.get('total_rows', 0):,}" for c in sorted_reg],
    textposition="outside",
))
fig_vol.update_layout(
    **PLOTLY_LAYOUT,
    title="총 경매 기록 수",
    height=max(400, len(registry) * 28),
    xaxis_title="건수",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# ── Quality scatter ──────────────────────────────────────────────────
st.subheader("데이터 품질 매트릭스")
st.caption("X축: 변동계수 (CV) — 높을수록 가격 변동 큼 | Y축: Lag-1 자기상관 — 높을수록 예측 용이")
st.caption("원 크기: 데이터 건수 비례 | 초록: 최신, 주황: 주의, 빨강: 중단")

fig_quality = quality_scatter(registry)
st.plotly_chart(fig_quality, use_container_width=True)
