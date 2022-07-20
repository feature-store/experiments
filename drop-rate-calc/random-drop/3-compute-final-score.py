from multiprocessing import Pool

import duckdb
import numpy as np
from sktime.performance_metrics.forecasting import mean_squared_scaled_error
from tqdm import tqdm
import yaml

config = yaml.safe_load(open("config.yaml"))
ground_truth = np.load(config["ground_truth_array_path"])
predictions = np.load(config["compact_prediction_output_path"])
conn = duckdb.connect(config["raw_db_path"], read_only=True)
conn.execute("PRAGMA enable_progress_bar")
num_windows_df = conn.execute(
    """
SELECT int_id, (timestamp[-1] - timestamp[1]+1)/288-2 as num_windows
  FROM readings
ORDER BY int_id ASC
"""
).df()

assert ground_truth.shape == predictions.shape

masked_ground_truth = ground_truth.copy()
np.place(masked_ground_truth, mask=np.isnan(predictions), vals=np.nan)

extra_nans_per_row = (
    np.isnan(predictions).sum() - np.isnan(ground_truth).sum()
) / 275077
assert 830 < extra_nans_per_row < 870, extra_nans_per_row

y_true = np.nan_to_num(masked_ground_truth, 0.0)
y_pred = np.nan_to_num(predictions, 0.0)
assert y_true.shape == y_pred.shape


def error_func(i):
    y_true_, y_pred_ = y_true[i], y_pred[i]
    return mean_squared_scaled_error(y_true_, y_pred_, sp=288, y_train=y_true_)


with Pool() as p:
    msse = list(
        tqdm(p.imap(error_func, range(len(y_true)), chunksize=64), total=len(y_true))
    )

final_score = (msse / num_windows_df["num_windows"]).mean()
print(final_score)
