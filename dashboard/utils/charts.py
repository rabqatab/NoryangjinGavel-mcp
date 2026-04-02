"""Reusable Plotly chart builders for the Noryangjin dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import MODEL_COLORS, PLOTLY_LAYOUT


def _base_layout(**overrides) -> dict:
    layout = {**PLOTLY_LAYOUT, **overrides}
    return layout


def timeseries_chart(
    df: pd.DataFrame,
    date_col: str = "trade_date_dt",
    price_col: str = "price_avg",
    title: str = "",
    show_ma: bool = False,
    show_volume: bool = True,
) -> go.Figure:
    """Interactive time series with optional MA overlays and volume bars."""
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2] if show_volume else [1.0],
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_col], y=df[price_col],
            name="일평균가", mode="lines",
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=1, col=1,
    )

    if show_ma:
        for span, color, name in [(7, "#ff7f0e", "7일 이동평균"), (30, "#d62728", "30일 이동평균")]:
            ma = df[price_col].rolling(span, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=df[date_col], y=ma,
                    name=name, mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                ),
                row=1, col=1,
            )

    if show_volume and "quantity" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df[date_col], y=df["quantity"],
                name="거래량", marker_color="rgba(100,100,100,0.3)",
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="거래량", row=2, col=1)

    fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        row=2 if show_volume else 1, col=1,
    )
    fig.update_layout(
        **_base_layout(title=title, height=500 if show_volume else 400, showlegend=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def seasonality_heatmap(df: pd.DataFrame, date_col: str = "trade_date_dt", price_col: str = "price_avg", title: str = "월별 계절성") -> go.Figure:
    """Year x Month heatmap of average prices."""
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    pivot = df.groupby(["year", "month"])[price_col].mean().unstack(fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{m}월" for m in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="YlOrRd",
        colorbar=dict(title="평균가 (원)"),
    ))
    fig.update_layout(**_base_layout(title=title, height=max(300, len(pivot) * 20)))
    return fig


def mape_bar_chart(
    config_ids: list[str],
    mape_values: list[float],
    models: list[str] | None = None,
    title: str = "MAPE 비교",
) -> go.Figure:
    """Horizontal bar chart of MAPE values per config."""
    colors = [MODEL_COLORS.get(m, "#999999") for m in models] if models else "#1f77b4"

    fig = go.Figure(go.Bar(
        y=config_ids, x=mape_values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in mape_values],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(title=title, height=max(400, len(config_ids) * 28)),
        xaxis_title="MAPE (%)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def model_heatmap(
    matrix: list[list[float]],
    config_ids: list[str],
    model_names: list[str],
    title: str = "모델별 MAPE 히트맵",
) -> go.Figure:
    """Configs (y) x Models (x) MAPE heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=model_names,
        y=config_ids,
        colorscale="RdYlGn_r",
        zmin=0, zmax=50,
        colorbar=dict(title="MAPE (%)"),
        text=[[f"{v:.1f}" if v < 900 else "-" for v in row] for row in matrix],
        texttemplate="%{text}",
    ))
    fig.update_layout(**_base_layout(
        title=title,
        height=max(500, len(config_ids) * 30),
        xaxis=dict(side="top"),
    ))
    return fig


def feature_importance_chart(
    importance: dict[str, float],
    top_n: int = 20,
    title: str = "특성 중요도 상위 20",
) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in reversed(sorted_items)]
    values = [x[1] for x in reversed(sorted_items)]

    fig = go.Figure(go.Bar(
        y=names, x=values,
        orientation="h",
        marker_color="#ff7f0e",
    ))
    fig.update_layout(**_base_layout(title=title, height=max(400, top_n * 25)))
    return fig


def quality_scatter(
    configs: list[dict],
    title: str = "데이터 품질 매트릭스",
) -> go.Figure:
    """CV (x) vs Lag-1 (y) scatter, size=rows, color=freshness."""
    cvs = [c.get("cv", 0) for c in configs]
    lag1s = [c.get("lag1", 0) for c in configs]
    sizes = [max(8, min(40, c.get("total_rows", 1000) / 500)) for c in configs]
    labels = [c.get("id", "") for c in configs]
    recent = [c.get("recent_days_2025", 0) for c in configs]

    colors = []
    for r in recent:
        if r >= 100:
            colors.append("#2ca02c")
        elif r >= 1:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    fig = go.Figure(go.Scatter(
        x=cvs, y=lag1s,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=1, color="white")),
        text=labels,
        textposition="top center",
        textfont=dict(size=9),
    ))
    fig.add_hline(y=0.4, line_dash="dash", line_color="gray", annotation_text="Lag-1 = 0.4")
    fig.add_vline(x=0.8, line_dash="dash", line_color="gray", annotation_text="CV = 0.8")
    fig.update_layout(**_base_layout(
        title=title, height=500,
        xaxis_title="변동계수 (CV)", yaxis_title="Lag-1 자기상관",
    ))
    return fig
