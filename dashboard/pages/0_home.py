"""Home page — KPI overview."""

import streamlit as st

from utils.constants import (
    BEST_OF_BREED,
    CATEGORY_MAP,
    CONFIG_IDS,
    PREMIUM_IDS,
    SASHIMI_IDS,
    STANDARD_IDS,
)
from utils.data_loader import load_config_registry

st.title("노량진수산시장 시세 예측 대시보드")
st.caption("노량진수산시장 경락시세 데이터 기반 가격 예측 모니터링 시스템")

# KPI metrics
registry = load_config_registry()
total_rows = sum(c.get("total_rows", 0) for c in registry) if registry else 0
avg_mape = sum(v["mape"] for v in BEST_OF_BREED.values()) / len(BEST_OF_BREED)
best_config = min(BEST_OF_BREED.items(), key=lambda x: x[1]["mape"])

k1, k2, k3, k4 = st.columns(4)
k1.metric("총 경매 기록", f"{total_rows:,}건")
k2.metric("모니터링 품목", f"{len(CONFIG_IDS)}개")
k3.metric("평균 MAPE", f"{avg_mape:.1f}%")
k4.metric("최고 성능", f"{best_config[1]['mape']:.1f}%", delta=best_config[0])

st.markdown("---")

# Best / Worst
col_best, col_worst = st.columns(2)

with col_best:
    st.subheader("성능 상위 5개 품목")
    for i, (cid, info) in enumerate(sorted(BEST_OF_BREED.items(), key=lambda x: x[1]["mape"])[:5], 1):
        cat = CATEGORY_MAP.get(cid, "")
        st.markdown(f"**{i}. {cid}** — MAPE **{info['mape']:.1f}%** ({info['model']}) `{cat}`")

with col_worst:
    st.subheader("성능 하위 5개 품목")
    for i, (cid, info) in enumerate(sorted(BEST_OF_BREED.items(), key=lambda x: x[1]["mape"], reverse=True)[:5], 1):
        cat = CATEGORY_MAP.get(cid, "")
        st.markdown(f"**{i}. {cid}** — MAPE **{info['mape']:.1f}%** ({info['model']}) `{cat}`")

st.markdown("---")

# Category Summary
st.subheader("카테고리별 요약")
cat_cols = st.columns(3)

for col, (cat_name, cat_ids) in zip(cat_cols, [
    ("회용 (횟감)", SASHIMI_IDS),
    ("일반", STANDARD_IDS),
    ("프리미엄 활어", PREMIUM_IDS),
]):
    cat_mapes = [BEST_OF_BREED[cid]["mape"] for cid in cat_ids if cid in BEST_OF_BREED]
    avg = sum(cat_mapes) / len(cat_mapes) if cat_mapes else 0
    best = min(cat_mapes) if cat_mapes else 0
    with col:
        st.metric(cat_name, f"평균 {avg:.1f}%", delta=f"최고 {best:.1f}%")
        st.caption(f"{len(cat_ids)}개 품목")

st.markdown("---")

# Freshness Warnings
inactive = [c for c in registry if c.get("recent_days_2025", 0) == 0]
if inactive:
    st.subheader("데이터 갱신 필요 품목")
    st.warning(f"{len(inactive)}개 품목의 2025년 데이터가 없습니다. 크롤러 설정을 확인해 주세요.")
    for c in inactive:
        st.markdown(f"- **{c['id']}** — 마지막 거래: {c.get('date_range', 'N/A').split('~')[-1].strip()}")
else:
    st.success("모든 품목의 데이터가 최신 상태입니다.")
