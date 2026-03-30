import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def setup():
    import marimo as mo
    import numpy as np
    from collections import Counter, defaultdict

    from eda_helpers import (
        load_all_data, load_top_species, cv, weighted_avg,
        pearson_corr, lag1_autocorr, classify_spec,
    )

    mo.md("# Aggregation EDA: Row Integration Viability")
    return mo, np, Counter, defaultdict, load_all_data, load_top_species, cv, weighted_avg, pearson_corr, lag1_autocorr, classify_spec


@app.cell
def load_data(load_all_data, load_top_species, mo):
    mo.md("## Data Loading")
    data = load_all_data()
    n_rows = len(data["trade_date"])
    top10 = load_top_species(data, 10)
    mo.md(f"Loaded **{n_rows:,}** rows. Top 10 species: {', '.join(top10)}")
    return data, n_rows, top10


if __name__ == "__main__":
    app.run()
