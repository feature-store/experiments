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
SEED = 42

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
os.makedirs(output_dir, exist_ok=True)


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

    sampled_df = df.sample(
        frac=float(os.environ.get("KEEP_FRAC_OVERRIDE", 0)) or config["keep_frac"],
        random_state=SEED,
    ).sort_values("start_idx")
    # ensure at least one window, we will use the first window here by default
    if len(sampled_df) == 0:
        sampled_df = df.iloc[:1, :]

    for _, row in sampled_df.iterrows():
        pred_start_idx = ground_truth_start_idx + row["start_idx"] + WINDOW_SIZE
        remaining_pred_size = ground_truth_end_idx - pred_start_idx
        if remaining_pred_size <= 0:
            continue
        forecast_to_fill = row["forecast_arr"][: remaining_pred_size + 1]
        pred_arr[
            pred_start_idx : pred_start_idx + remaining_pred_size + 1
        ] = forecast_to_fill

    np.save(os.path.join(output_dir, f"{int_id}.npy"), pred_arr)


with Pool() as pool:
    list(tqdm(pool.imap(generate_oracle, paths, chunksize=64), total=len(paths)))
