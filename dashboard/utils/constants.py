"""Shared constants for the Noryangjin dashboard."""

from __future__ import annotations

FOREIGN_KW = [
    "일본", "중국", "미국", "러시아", "캐나다", "노르웨이", "뉴질랜드", "대만", "칠레",
    "아르헨티나", "영국", "아일랜드", "온두라스", "북한", "(원양)", "인도", "인도네시아",
    "태국", "베트남", "필리핀", "호주", "스페인", "네덜란드", "페루", "모로코", "아프리카",
    "파키스탄", "라스팔마스", "포클랜드", "멕시코",
]

SPECIES_CONFIGS: list[dict] = [
    {"id": "넙치_활_kg_중", "species": "넙치", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "우럭_활_kg_중", "species": "우럭", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "방어_선_kg_중_dom", "species": "방어", "state": "선", "pkg": "kg", "spec": "중", "domestic": True},
    {"id": "참돔_활_kg_중_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"id": "농어_활_kg_중_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"id": "도다리_활_kg_중", "species": "도다리", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "감성돔_활_kg_중_dom", "species": "감성돔", "state": "활", "pkg": "kg", "spec": "중", "domestic": True},
    {"id": "감숭어_활_kg_중", "species": "감숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "참숭어_활_kg_중", "species": "참숭어", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "쭈꾸미_선_box_중_dom", "species": "쭈꾸미", "state": "선", "pkg": "box", "spec": "중", "domestic": True},
    {"id": "민어_선_SP_중", "species": "민어", "state": "선", "pkg": "S/P", "spec": "중", "domestic": False},
    {"id": "깐굴_선_box_소", "species": "깐굴", "state": "선", "pkg": "box", "spec": "소", "domestic": False},
    {"id": "바위굴_활_box_대", "species": "바위굴", "state": "활", "pkg": "box", "spec": "대", "domestic": False},
    {"id": "수꽃게_활_kg_중", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "암꽃게_활_kg_중", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "중", "domestic": False},
    {"id": "수꽃게_활_kg_대", "species": "수꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False},
    {"id": "암꽃게_활_kg_대", "species": "암꽃게", "state": "활", "pkg": "kg", "spec": "대", "domestic": False},
    {"id": "넙치_활_kg_2미", "species": "넙치", "state": "활", "pkg": "kg", "spec": "2미", "domestic": False},
    {"id": "참돔_활_kg_2미_dom", "species": "참돔", "state": "활", "pkg": "kg", "spec": "2미", "domestic": True},
    {"id": "농어_활_kg_1미_dom", "species": "농어", "state": "활", "pkg": "kg", "spec": "1미", "domestic": True},
]

CONFIG_IDS = [c["id"] for c in SPECIES_CONFIGS]

# Categories
SASHIMI_IDS = [
    "넙치_활_kg_중", "우럭_활_kg_중", "방어_선_kg_중_dom",
    "참돔_활_kg_중_dom", "농어_활_kg_중_dom", "도다리_활_kg_중", "감성돔_활_kg_중_dom",
]
STANDARD_IDS = [
    "감숭어_활_kg_중", "참숭어_활_kg_중", "쭈꾸미_선_box_중_dom", "민어_선_SP_중",
    "깐굴_선_box_소", "바위굴_활_box_대", "수꽃게_활_kg_중", "암꽃게_활_kg_중",
]
PREMIUM_IDS = [
    "수꽃게_활_kg_대", "암꽃게_활_kg_대", "넙치_활_kg_2미", "참돔_활_kg_2미_dom", "농어_활_kg_1미_dom",
]

CATEGORY_MAP = {cid: "회용 (횟감)" for cid in SASHIMI_IDS}
CATEGORY_MAP.update({cid: "일반" for cid in STANDARD_IDS})
CATEGORY_MAP.update({cid: "프리미엄 활어" for cid in PREMIUM_IDS})

# Best-of-breed results (from docs/15_prediction_config_registry.md)
BEST_OF_BREED: dict[str, dict] = {
    "바위굴_활_box_대": {"mape": 2.1, "model": "Transformer-Q"},
    "쭈꾸미_선_box_중_dom": {"mape": 6.8, "model": "GRU-Q"},
    "깐굴_선_box_소": {"mape": 7.6, "model": "Transformer-Q"},
    "암꽃게_활_kg_대": {"mape": 8.2, "model": "GRU-Q"},
    "수꽃게_활_kg_대": {"mape": 10.2, "model": "GRU-Q"},
    "수꽃게_활_kg_중": {"mape": 10.2, "model": "Transformer-Q"},
    "암꽃게_활_kg_중": {"mape": 10.3, "model": "Transformer-Q"},
    "넙치_활_kg_중": {"mape": 11.1, "model": "v11 LightGBM"},
    "감성돔_활_kg_중_dom": {"mape": 12.5, "model": "GRU-Q"},
    "농어_활_kg_중_dom": {"mape": 12.7, "model": "GRU-Q"},
    "농어_활_kg_1미_dom": {"mape": 13.0, "model": "v11 LightGBM"},
    "우럭_활_kg_중": {"mape": 14.7, "model": "TFT"},
    "도다리_활_kg_중": {"mape": 15.0, "model": "Transformer-Q"},
    "방어_선_kg_중_dom": {"mape": 15.6, "model": "TFT"},
    "참돔_활_kg_중_dom": {"mape": 16.2, "model": "Transformer-Q"},
    "참숭어_활_kg_중": {"mape": 17.1, "model": "Transformer-Q"},
    "참돔_활_kg_2미_dom": {"mape": 17.7, "model": "v11 LightGBM"},
    "넙치_활_kg_2미": {"mape": 19.1, "model": "Transformer-Q"},
    "감숭어_활_kg_중": {"mape": 19.9, "model": "Transformer-Q"},
    "민어_선_SP_중": {"mape": 34.0, "model": "v11 LightGBM"},
}

MODEL_COLORS: dict[str, str] = {
    "LightGBM": "#1f77b4",
    "v11 LightGBM": "#1f77b4",
    "TFT": "#d62728",
    "GRU": "#2ca02c",
    "GRU-Q": "#2ca02c",
    "LSTM": "#9467bd",
    "BiLSTM+Attn": "#8c564b",
    "CNN-LSTM": "#e377c2",
    "CNN-LSTM-Q": "#e377c2",
    "Transformer": "#ff7f0e",
    "Transformer-Q": "#ff7f0e",
    "PatchTST": "#17becf",
}

PLOTLY_FONT = "Noto Sans CJK KR, Noto Sans CJK JP, sans-serif"

PLOTLY_LAYOUT = dict(
    font=dict(family=PLOTLY_FONT, size=13),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=50, b=50),
)
