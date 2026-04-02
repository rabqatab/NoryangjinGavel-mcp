"""Consumer price context — wholesale vs retail price gap."""

import plotly.graph_objects as go
import streamlit as st

from utils.constants import CONFIG_IDS, PLOTLY_LAYOUT, SPECIES_CONFIGS
from utils.data_loader import load_daily_prices

st.title("소비자 가격 안내")
st.caption("경락가(도매 경매가)와 실제 소비자가의 차이를 안내합니다")

# ── Price Chain Explainer ─────────────────────────────────────────
st.subheader("수산물 유통 구조")
st.markdown("""
이 대시보드의 가격은 **노량진수산시장 경락가(경매 낙찰가)**입니다.
소비자가 실제로 지불하는 가격은 유통 단계를 거치면서 **2~3배** 이상 높아집니다.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("경락가 (경매)", "기준가", help="이 대시보드에 표시되는 가격")
col2.metric("중도매인", "+10~20%", help="도매시장 내 중간상인 마진")
col3.metric("소매시장/마트", "+50~80%", help="대형마트, 재래시장 판매가")
col4.metric("횟집/식당", "+150~300%", help="회로 떠서 판매하는 가격")

st.markdown("---")

# ── Distribution Cost Ratio (해양수산부 data) ─────────────────────
st.subheader("유통 비용률 (해양수산부 실태조사)")
st.caption("소비자가 대비 유통 비용이 차지하는 비율. 높을수록 중간 마진이 큽니다.")

# Source: 해양수산부 2020 수산물 유통산업실태조사 + 2024 update
DISTRIBUTION_COST = {
    "넙치 (광어)": {"ratio": 70.0, "auction_kg": 14500, "category": "활어회"},
    "우럭": {"ratio": 62.0, "auction_kg": 10500, "category": "활어회"},
    "참돔": {"ratio": 63.0, "auction_kg": 15800, "category": "활어회"},
    "방어": {"ratio": 55.0, "auction_kg": 5600, "category": "선어회"},
    "농어": {"ratio": 58.0, "auction_kg": 14400, "category": "활어회"},
    "고등어": {"ratio": 68.5, "auction_kg": 4500, "category": "대중어"},
    "오징어": {"ratio": 52.0, "auction_kg": 23300, "category": "선어"},
    "삼치": {"ratio": 54.0, "auction_kg": 31800, "category": "선어"},
    "갈치": {"ratio": 58.0, "auction_kg": 50600, "category": "선어"},
    "꽃게 (수)": {"ratio": 55.0, "auction_kg": 14900, "category": "갑각류"},
    "꽃게 (암)": {"ratio": 55.0, "auction_kg": 26200, "category": "갑각류"},
    "전복": {"ratio": 48.0, "auction_kg": 32300, "category": "패류"},
    "굴 (깐굴)": {"ratio": 45.0, "auction_kg": 18600, "category": "패류"},
    "낙지": {"ratio": 50.0, "auction_kg": 15000, "category": "연체류"},
    "쭈꾸미": {"ratio": 48.0, "auction_kg": 34100, "category": "연체류"},
    "소라": {"ratio": 42.0, "auction_kg": 71600, "category": "패류"},
}

species_names = list(DISTRIBUTION_COST.keys())
ratios = [v["ratio"] for v in DISTRIBUTION_COST.values()]
auction_prices = [v["auction_kg"] for v in DISTRIBUTION_COST.values()]
categories = [v["category"] for v in DISTRIBUTION_COST.values()]

# Calculate estimated consumer price
consumer_prices = [
    int(auc / (1 - ratio / 100)) for auc, ratio in zip(auction_prices, ratios)
]
markups = [round(cp / ap, 1) for cp, ap in zip(consumer_prices, auction_prices)]

# Bar chart: auction vs consumer
fig = go.Figure()
fig.add_trace(go.Bar(
    name="경락가 (도매 경매)",
    y=species_names, x=auction_prices,
    orientation="h",
    marker_color="#1f77b4",
    text=[f"{p:,}원" for p in auction_prices],
    textposition="inside",
))
fig.add_trace(go.Bar(
    name="추정 소비자가 (소매)",
    y=species_names, x=consumer_prices,
    orientation="h",
    marker_color="#ff7f0e",
    text=[f"{p:,}원 (x{m})" for p, m in zip(consumer_prices, markups)],
    textposition="inside",
))
fig.update_layout(
    **PLOTLY_LAYOUT,
    title="경락가 vs 추정 소비자가 (kg당)",
    barmode="group",
    height=max(500, len(species_names) * 40),
    xaxis_title="가격 (원/kg)",
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Markup Table ──────────────────────────────────────────────────
st.subheader("품목별 유통 마진 상세")

import pandas as pd

rows = []
for name, info in DISTRIBUTION_COST.items():
    auc = info["auction_kg"]
    ratio = info["ratio"]
    consumer = int(auc / (1 - ratio / 100))
    rows.append({
        "품목": name,
        "분류": info["category"],
        "경락가 (원/kg)": f"{auc:,}",
        "유통비용률": f"{ratio:.0f}%",
        "추정 소비자가": f"{consumer:,}원",
        "배수": f"x{consumer/auc:.1f}",
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("출처: 해양수산부 수산물 유통산업실태조사 (2020, 2024), 경락가는 본 대시보드 2024-2025 평균")

st.markdown("---")

# ── Channel Comparison ────────────────────────────────────────────
st.subheader("구매 채널별 예상 가격")
st.caption("같은 어종이라도 구매 채널에 따라 가격이 크게 달라집니다")

selected = st.selectbox("어종 선택", list(DISTRIBUTION_COST.keys()), key="chain_species")
info = DISTRIBUTION_COST[selected]
base = info["auction_kg"]

channels = {
    "경락가 (경매 낙찰)": 1.0,
    "도매시장 중도매인": 1.15,
    "노량진 소매상 (직접구매)": 1.4,
    "대형마트 (이마트/홈플러스)": 1.8,
    "재래시장 어물전": 1.6,
    "온라인 (쿠팡/마켓컬리)": 1.7,
    "횟집 (회로 제공)": 3.0,
    "일식집 (코스 요리)": 5.0,
}

ch_names = list(channels.keys())
ch_prices = [int(base * mult) for mult in channels.values()]
ch_mults = list(channels.values())

colors = ["#1f77b4"] + ["#aec7e8"] * 2 + ["#ff7f0e"] * 3 + ["#d62728"] * 2

fig2 = go.Figure(go.Bar(
    y=ch_names, x=ch_prices,
    orientation="h",
    marker_color=colors,
    text=[f"{p:,}원 (x{m:.1f})" for p, m in zip(ch_prices, ch_mults)],
    textposition="outside",
))
fig2.update_layout(
    **PLOTLY_LAYOUT,
    title=f"{selected} — 채널별 예상 가격 (kg당)",
    height=400,
    xaxis_title="가격 (원/kg)",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Sashimi Conversion ────────────────────────────────────────────
st.subheader("횟감 환산 안내")
st.caption("활어 1kg에서 실제 먹을 수 있는 회의 양과 가격")

st.markdown("""
| 항목 | 값 | 설명 |
|---|---|---|
| **활어 1kg** | 100% | 도매 경매에서 거래되는 단위 |
| **회 수율** | 35~45% | 뼈, 내장, 머리 제외 (어종별 상이) |
| **실제 회** | 350~450g | 1kg 활어에서 나오는 회 양 |
| **횟집 1인분** | 200~300g | 일반적인 1인 모듬회 기준 |
""")

sashimi_species = {
    "넙치 (광어)": {"yield": 0.42, "price_kg": 14500},
    "우럭": {"yield": 0.38, "price_kg": 10500},
    "참돔": {"yield": 0.40, "price_kg": 15800},
    "방어": {"yield": 0.45, "price_kg": 5600},
    "농어": {"yield": 0.40, "price_kg": 14400},
    "도다리": {"yield": 0.35, "price_kg": 13400},
}

sashi_rows = []
for name, info in sashimi_species.items():
    auction_g = info["price_kg"]  # per kg
    yield_rate = info["yield"]
    sashimi_per_kg = yield_rate * 1000  # grams of sashimi from 1kg fish
    cost_per_100g = int(auction_g / (sashimi_per_kg / 100))  # auction cost per 100g sashimi
    restaurant_per_100g = cost_per_100g * 3  # typical restaurant markup
    serving_g = 250  # average serving
    serving_auction = int(auction_g * serving_g / sashimi_per_kg)
    serving_restaurant = serving_auction * 3

    sashi_rows.append({
        "어종": name,
        "회 수율": f"{yield_rate*100:.0f}%",
        "경락가 (kg)": f"{auction_g:,}원",
        "회 100g 원가": f"{cost_per_100g:,}원",
        "1인분 원가 (250g)": f"{serving_auction:,}원",
        "횟집 예상가 (1인)": f"~{serving_restaurant:,}원",
    })

st.dataframe(pd.DataFrame(sashi_rows), use_container_width=True, hide_index=True)

st.info(
    "위 경락가는 도매 경매 낙찰가로, 일반 소비자가 직접 구매할 수 있는 가격이 아닙니다. "
    "노량진시장 소매상에서는 경락가의 약 1.3~1.5배, "
    "대형마트에서는 1.7~2.0배, 횟집에서는 2.5~4.0배 수준입니다."
)

st.markdown("---")

# ── KAMIS Retail Reference (API data) ─────────────────────────────
st.subheader("KAMIS 공식 소매가격 (참고)")
st.caption("한국농수산식품유통공사(KAMIS) Open API에서 제공하는 수산물 소매가격. API 제공 품목이 제한적입니다.")

# KAMIS API provides retail data for limited seafood items
# Data source: yearlySalesList, category 600, yearly avg 2024
KAMIS_RETAIL = {
    "마른멸치 (대멸)": {"wholesale": 13991, "retail": 23190, "unit": "1kg", "code": 638},
    "마른미역": {"wholesale": 13984, "retail": 30930, "unit": "1kg", "code": 642},
    "굴": {"wholesale": 13899, "retail": 21165, "unit": "1kg", "code": 644},
    "가리비 (홍가리비)": {"wholesale": None, "retail": 7868, "unit": "1kg", "code": 659},
    "건다시마": {"wholesale": None, "retail": 37470, "unit": "1kg", "code": 660},
    "홍합 (깐)": {"wholesale": None, "retail": 27440, "unit": "1kg", "code": 658},
    "홍합 (안깐)": {"wholesale": None, "retail": 4062, "unit": "1kg", "code": 658},
}

kamis_rows = []
for name, info in KAMIS_RETAIL.items():
    ws = info["wholesale"]
    rt = info["retail"]
    markup = f"x{rt/ws:.1f}" if ws else "-"
    kamis_rows.append({
        "품목": name,
        "중도매 판매가": f"{ws:,}원" if ws else "-",
        "소매가격": f"{rt:,}원",
        "배수": markup,
        "출처": "KAMIS API (2024 평균)",
    })

st.dataframe(pd.DataFrame(kamis_rows), use_container_width=True, hide_index=True)

st.warning(
    "KAMIS Open API는 수산물 소매가격을 7개 품목만 제공합니다. "
    "주요 활어(넙치, 우럭, 방어, 참돔 등)의 소매가격은 API에서 제공되지 않아 "
    "해양수산부 유통비용률 기반으로 추정합니다."
)

st.markdown("---")
st.caption("출처: 해양수산부 수산물 유통산업실태조사, KAMIS Open API (kamis.or.kr)")
st.caption("유통비용률은 추정치이며, 실제 가격은 시기·지역·품질에 따라 다를 수 있습니다.")
