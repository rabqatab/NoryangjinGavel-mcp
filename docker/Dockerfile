FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    "pytorch-forecasting>=1.0.0,<1.2" \
    "lightning>=2.0,<2.5" \
    optuna==4.1.0 \
    lightgbm==4.5.0 \
    pyarrow==18.1.0 \
    vmdpy==0.2 \
    scikit-learn==1.6.0 \
    statsmodels==0.14.4 \
    matplotlib==3.10.0

# Copy project files
COPY scripts/ scripts/
COPY data/parquet/ data/parquet/
COPY crawler/species_inventory.json crawler/

# Default: run TFT training
CMD ["python", "scripts/train_tft.py"]
