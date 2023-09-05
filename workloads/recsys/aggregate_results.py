import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset, read_config, log_dataset, use_plots, log_plots

from tqdm import tqdm
import os

experiment = "ml-1m"
dataset_dir = use_dataset(experiment)
result_dir = use_results(experiment, download=False)

# experiment parameters
split = 0.75
#updates_per_ts = [0.5, 0.25, 0.2, 1, 2, 3, 4, 5, 8, 10000, None]
updates_per_ts = [0.5, 0.25, 0.2, 1, 2, 3, 4, 5, 8, None, 10000] #, 3, 4, 5, 8]
ts_factor = [60] #, 60*60, 60*60*24]
policies = ["random", "total_error_cold", "query_proportional", "max_pending", "min_past", "round_robin", "batch"] #, "last_query"]
dist = ["exponential", "gaussian", None]
limit = 100000

stream_df = pd.read_csv(f'{dataset_dir}/stream_{split}.csv')
start_ts = stream_df.timestamp.min()
stream_df.timestamp = stream_df.timestamp.apply(lambda ts: ts - start_ts)


# collect baselines 
results = []

for baseline in [None, 10000]:
    for t in ts_factor:
        for d in dist:

            if d:
                update_df = pd.read_csv(f"{result_dir}/round_robin_{baseline}_{t}_split_{split}_dist_{d}_updates.csv")
                df = pd.read_csv(f"{result_dir}/round_robin_{baseline}_{t}_split_{split}_dist_{d}_results.csv")
            else:
                update_df = pd.read_csv(f"{result_dir}/round_robin_{baseline}_{t}_split_{split}_updates.csv")
                df = pd.read_csv(f"{result_dir}/round_robin_{baseline}_{t}_split_{split}_results.csv")

            print(len(df.index))
            df = df.iloc[:limit]

            timestamp = df.timestamp.max()
            update_df = update_df[update_df["time"] < timestamp]

            error = mean_squared_error(df.y_pred, df.y_true)
            total_updates = len(update_df.index)

            if baseline is None:
                results.append([0, error, total_updates, t, df.timestamp.max(), d])
            else:
                results.append([baseline, error, total_updates, t, df.timestamp.max(), d])

baseline_results_df = pd.DataFrame(results, columns=["updates", "error", "total_updates", "ts_factor", "max_ts", "dist"])

# collect all results 
updates_df = pd.DataFrame()
df_all = pd.DataFrame()
results = []

for p in policies:
    for u in updates_per_ts:
        for t in ts_factor:

            for d in dist:

                if d:
                    path = f"{result_dir}/{p}_{u}_{t}_split_{split}_dist_{d}"
                else:
                    path = f"{result_dir}/{p}_{u}_{t}_split_{split}"

                if not os.path.exists(f"{path}_updates.csv"):
                    print("missing", f"{path}_updates.csv")
                    continue

                update_df = pd.read_csv(f"{path}_updates.csv")
                df = pd.read_csv(f"{path}_results.csv")


                if limit is not None:
                    if len(df.index) <= limit:
                        print("max index", len(df.index), f"{result_dir}/{p}_{u}_{t}_split_{split}_results.csv")
                        continue

                    # filter time
                    df = df.iloc[:limit]
                    timestamp = df.timestamp.max()
                    update_df = update_df[update_df["time"] < timestamp]

                print(p, u, t, df.timestamp.max(), update_df["time"].max())

                df["policy"] = p
                df["updates"] = u
                df["ts_factor"] = t
                df["dist"] = d
                update_df["updates"] = u
                update_df["ts_factor"] = t
                update_df["dist"] = d

                # threshold predictions
                df["y_pred"][df["y_pred"] > 5] = 5
                df["y_pred"][df["y_pred"] < 1] = 1

                #print(df.y_pred)
                #print(df.y_true)

                error = mean_squared_error(df.y_pred, df.y_true)

                #print(update_df.time.value_counts())


                total_updates = len(update_df.index)
                print(total_updates, len(df.index), error)
                results.append([u , p, error, total_updates, t, df.timestamp.max(), d])
                updates_df = pd.concat([updates_df, update_df])
                df_all = pd.concat([df_all, df])

results_df = pd.DataFrame(results, columns=["updates", "policy", "error", "total_updates", "ts_factor", "max_ts", "dist"])


# save results
plots_dir = use_plots("ml-1m-dist")
df_all.to_csv(f"{plots_dir}/predictions.csv")
results_df.to_csv(f"{plots_dir}/results.csv")
baseline_results_df.to_csv(f"{plots_dir}/baseline.csv")
updates_df.to_csv(f"{plots_dir}/updates.csv")
log_plots("ml-1m-dist")
