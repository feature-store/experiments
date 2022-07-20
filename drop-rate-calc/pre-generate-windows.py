import os
import warnings
from multiprocessing import Pool

import duckdb
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from tqdm import tqdm

WINDOW_SIZE = 864
SLIDE_SIZE = 288
TOTAL_LENGTH = 8639
FINAL_IDX = 8638
DEBUG_MODE = False
RESULT_DIR = "/data/azure-windows"
SAMPLE_SIZE = 275077


conn = duckdb.connect(
    "/home/ubuntu/azure_long_series_with_ts_array.duckdb", read_only=True
)
conn.execute("PRAGMA enable_progress_bar")
cursor = conn.execute(
    """
SELECT int_id, avg_cpu
  FROM readings
-- USING SAMPLE 1000
"""
)


def fit(arr):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return (
            STLForecast(
                pd.Series(arr).interpolate(),
                ARIMA,
                model_kwargs=dict(order=(1, 1, 0), trend="t"),
                period=12 * 24,  # 5 min timestamp interval, period of one day
            )
            .fit()
            .forecast(7775)
            .values
        )  # return array


def window(arr):
    num_windows = len(arr) // SLIDE_SIZE - 2
    start_idx_to_window = {
        SLIDE_SIZE * i: arr[SLIDE_SIZE * i : SLIDE_SIZE * i + WINDOW_SIZE]
        for i in range(num_windows)
    }
    if DEBUG_MODE:
        for w in start_idx_to_window.values():
            assert len(w) == WINDOW_SIZE
    return start_idx_to_window


def map_row(row):
    int_id, avg_cpu = row
    entry = [
        {
            "int_id": int_id,
            "start_idx": k,
            "window_arr": np.array(v, dtype="float32"),
            "forecast_arr": fit(v).astype("float32"),
        }
        for k, v in window(avg_cpu).items()
    ]
    df = pd.DataFrame(entry)
    df.to_parquet(os.path.join(RESULT_DIR, f"{int_id}.parquet"))


with Pool() as pool:
    list(tqdm(pool.imap(map_row, cursor.fetchall()), total=SAMPLE_SIZE))
