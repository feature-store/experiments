from glob import glob
from multiprocessing import Pool
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import duckdb
import yaml

WINDOW_SIZE = 864
TOTAL_LENGTH = 8639
SLIDE_SIZE = 288

config = yaml.safe_load(open("config.yaml"))

paths = glob(config["generated_windows_glob_path"])
output_dir = config["materialized_prediction_output_path"]
conn = duckdb.connect(config["raw_db_path"], read_only=True)
conn.execute("PRAGMA enable_progress_bar")
idx_df = (
    conn.execute(
        "select int_id, timestamp[1] as start_idx, timestamp[-1] as end_idx from readings order by int_id"
    )
    .fetch_df()
    .set_index("int_id")
)


def generate_oracle(path):
    df = pd.read_parquet(path)

    int_id = int(os.path.split(path)[1].replace(".parquet", ""))
    range_series = idx_df.loc[int_id]
    ground_truth_start_idx, ground_truth_end_idx = (
        range_series["start_idx"],
        range_series["end_idx"],
    )
    pred_arr = np.empty(TOTAL_LENGTH, dtype="float32")
    pred_arr.fill(np.nan)

    for _, row in df.iterrows():
        pred_start_idx = ground_truth_start_idx + row["start_idx"] + WINDOW_SIZE
        remaining_pred_size = ground_truth_end_idx - pred_start_idx
        if remaining_pred_size <= 0:
            continue
        forecast_to_fill = row["forecast_arr"][: remaining_pred_size + 1]
        pred_arr[
            pred_start_idx : pred_start_idx + remaining_pred_size + 1
        ] = forecast_to_fill

    arr_data_length = len(pred_arr) - np.isnan(pred_arr).sum()
    expected_arr_data_length = ground_truth_end_idx + 1 - ground_truth_start_idx
    assert arr_data_length == expected_arr_data_length - WINDOW_SIZE

    np.save(os.path.join(output_dir, f"{int_id}.npy"), pred_arr)


with Pool() as pool:
    list(tqdm(pool.imap(generate_oracle, paths, chunksize=256), total=len(paths)))
