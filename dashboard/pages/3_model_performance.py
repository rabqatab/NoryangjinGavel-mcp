"""Model performance comparison page."""

import streamlit as st

from utils.charts import feature_importance_chart, model_heatmap
from utils.constants import BEST_OF_BREED, CONFIG_IDS, MODEL_COLORS, PLOTLY_LAYOUT
from utils.data_loader import (
    get_summary_image,
    load_dl_results,
    load_loss_comparison,
    load_v11_results,
)

st.title("모델 성능 비교")
st.caption("CPU (LightGBM v11) 및 GPU (DL 7종) 모델 성능 종합 비교")

tab1, tab2, tab3, tab4 = st.tabs([
    "MAPE 히트맵", "최고 모델 순위", "특성 중요도", "손실 함수 비교",
])

# ── Tab 1: MAPE Heatmap ──────────────────────────────────────────────
with tab1:
    st.subheader("모델별 MAPE 히트맵")
    st.caption("값이 낮을수록 예측 정확도가 높습니다 (초록). 빨강은 성능이 낮은 모델입니다.")

    dl = load_dl_results()
    results_pp = dl.get("results_preprocessing", {})

    if results_pp:
        DL_MODELS = ["GRU", "LSTM", "BiLSTM+Attn", "CNN-LSTM", "Transformer", "PatchTST",
                     "GRU+VMD", "LSTM+VMD", "Transformer+VMD", "TFT"]
        matrix = []
        valid_configs = []

        for cid in CONFIG_IDS:
            if cid not in results_pp:
                continue
            row = []
            for model in DL_MODELS:
                mape = results_pp[cid].get(model, {}).get("mape", 999)
                row.append(mape)
            matrix.append(row)
            valid_configs.append(cid)

        if matrix:
            fig = model_heatmap(matrix, valid_configs, DL_MODELS)
            st.plotly_chart(fig, use_container_width=True)
    else:
        img = get_summary_image("model_heatmap.png")
        if img:
            st.image(str(img), use_container_width=True)
        else:
            st.info("DL 결과 데이터가 없습니다.")

# ── Tab 2: Best-of-breed ranking ─────────────────────────────────────
with tab2:
    st.subheader("품목별 최고 모델 순위")

    import plotly.graph_objects as go

    sorted_bob = sorted(BEST_OF_BREED.items(), key=lambda x: x[1]["mape"])
    config_ids = [x[0] for x in sorted_bob]
    mapes = [x[1]["mape"] for x in sorted_bob]
    models = [x[1]["model"] for x in sorted_bob]
    colors = [MODEL_COLORS.get(m, "#999") for m in models]

    fig = go.Figure(go.Bar(
        y=config_ids, x=mapes,
        orientation="h",
        marker_color=colors,
        text=[f"{m:.1f}% ({mdl})" for m, mdl in zip(mapes, models)],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="품목별 최고 MAPE (낮을수록 좋음)",
        height=max(500, len(config_ids) * 30),
        xaxis_title="MAPE (%)",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # CPU vs GPU comparison
    st.subheader("CPU vs GPU 모델 비교")
    st.caption("LightGBM (CPU) 최고 성능 vs DL (GPU) 최고 성능")

    v11 = load_v11_results()
    v11_results = v11.get("results", [])
    v11_best = {}
    for r in v11_results:
        sp = r.get("species", "")
        mape = r.get("mape_best", r.get("mape", 999))
        if sp not in v11_best or mape < v11_best[sp]:
            v11_best[sp] = mape

    dl_best = {}
    if results_pp:
        for cid, models_dict in results_pp.items():
            best = min(
                (m.get("mape", 999) for m in models_dict.values() if isinstance(m, dict)),
                default=999,
            )
            if best < 900:
                dl_best[cid] = best

    # Show side-by-side for configs that have both
    compare_data = []
    for cid in CONFIG_IDS:
        cpu = v11_best.get(cid, None)
        gpu = dl_best.get(cid, None)
        if cpu is not None and gpu is not None:
            compare_data.append({"품목": cid, "CPU (LightGBM)": cpu, "GPU (DL)": gpu})

    if compare_data:
        import pandas as pd
        df_compare = pd.DataFrame(compare_data)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=df_compare["품목"], x=df_compare["CPU (LightGBM)"],
            name="CPU (LightGBM)", orientation="h", marker_color="#1f77b4",
        ))
        fig2.add_trace(go.Bar(
            y=df_compare["품목"], x=df_compare["GPU (DL)"],
            name="GPU (DL 최고)", orientation="h", marker_color="#ff7f0e",
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            title="CPU vs GPU MAPE 비교",
            height=max(400, len(compare_data) * 30),
            xaxis_title="MAPE (%)",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Feature Importance ─────────────────────────────────────────
with tab3:
    st.subheader("특성 중요도 분석")

    selected = st.selectbox("품목 선택", CONFIG_IDS, key="fi_config")

    v11 = load_v11_results()
    v11_results = v11.get("results", [])

    importance = None
    for r in v11_results:
        if r.get("species", "") == selected and r.get("importance"):
            importance = r["importance"]
            break

    if importance:
        top_n = st.slider("표시할 특성 수", 10, 40, 20)
        fig = feature_importance_chart(importance, top_n=top_n, title=f"{selected} — 특성 중요도 상위 {top_n}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        img = get_summary_image("feature_importance_top20.png")
        if img:
            st.image(str(img), caption="전체 평균 특성 중요도", use_container_width=True)
        else:
            st.info("해당 품목의 특성 중요도 데이터가 없습니다.")

# ── Tab 4: Loss Function Comparison ───────────────────────────────────
with tab4:
    st.subheader("손실 함수별 성능 비교")
    st.caption("MSE, MAE, MAPE, sMAPE, Huber, LogCosh 6가지 손실 함수 비교")

    loss_data = load_loss_comparison()
    loss_results = loss_data.get("results", {})

    if loss_results:
        selected_config = st.selectbox("품목 선택", list(loss_results.keys()), key="loss_config")
        config_loss = loss_results.get(selected_config, {})

        if config_loss:
            import plotly.graph_objects as go

            loss_names = ["MSE", "MAE", "MAPE", "sMAPE", "Huber", "LogCosh"]
            model_names = list(config_loss.keys())

            fig = go.Figure()
            for model_name in model_names:
                model_losses = config_loss[model_name]
                mapes = [model_losses.get(ln, {}).get("mape", None) for ln in loss_names]
                fig.add_trace(go.Bar(
                    name=model_name, x=loss_names, y=mapes,
                    text=[f"{m:.1f}%" if m else "-" for m in mapes],
                    textposition="outside",
                ))

            fig.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                title=f"{selected_config} — 손실 함수별 MAPE",
                xaxis_title="손실 함수",
                yaxis_title="MAPE (%)",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        img = get_summary_image("loss_function_comparison.png")
        if img:
            st.image(str(img), use_container_width=True)
        else:
            st.info("손실 함수 비교 데이터가 없습니다.")
