"""Noryangjin Fish Market Price Prediction Dashboard — Entry point."""

import streamlit as st

st.set_page_config(
    page_title="노량진수산시장 시세 예측",
    page_icon="\U0001F41F",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.constants import CONFIG_IDS

# ── Shared session state ─────────────────────────────────────────────
if "selected_config" not in st.session_state:
    st.session_state["selected_config"] = CONFIG_IDS[0]

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("노량진수산시장")
st.sidebar.caption("시세 예측 대시보드")
st.sidebar.markdown("---")

st.sidebar.selectbox(
    "품목 선택 (전체 페이지 공유)",
    CONFIG_IDS,
    key="selected_config",
)

# ── Navigation with Korean labels ────────────────────────────────────
pages = [
    st.Page("pages/0_home.py", title="홈", icon="\U0001F3E0", default=True),
    st.Page("pages/1_price_trends.py", title="시세 현황", icon="\U0001F4C8"),
    st.Page("pages/2_predictions.py", title="예측 결과", icon="\U0001F52E"),
    st.Page("pages/3_model_performance.py", title="모델 성능", icon="\U0001F4CA"),
    st.Page("pages/4_data_health.py", title="데이터 건강", icon="\U0001F3E5"),
    st.Page("pages/5_price_chain.py", title="소비자 가격", icon="\U0001F4B0"),
]

pg = st.navigation(pages)
pg.run()
