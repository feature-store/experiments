from collections import defaultdict
import json
import pickle
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import msgpack
import pandas as pd
import seaborn as sns
from absl import app, flags
from sktime.performance_metrics.forecasting import mean_squared_scaled_error

sns.set(style="whitegrid", palette="muted")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "experiment_dir", None, "point to experiment_timestamp directory", required=True
)
flags.DEFINE_integer(
    "send_rate_per_key", None, "records per seconds for each key", required=True
)
flags.DEFINE_string(
    "oracle_csv_glob_path",
    "/home/ubuntu/experiments/stl/offline/result/offline_1_slide/plan_eval/oracle_key_A4Benchmark-TS*.csv",
    "glob oracle csv genreated by offline eval",
)
flags.DEFINE_string(
    "oracle_csv_extraction_regex",
    ".*TS(\d+).csv",
    "regex to extract the key from path name",
)
flags.DEFINE_bool(
    "is_timely_result", False, "control whether to use timely specific dump processing"
)


def read_system_metrics_df(experiment_dir) -> pd.DataFrame:
    path = f"{experiment_dir}/snapshots.jsonl"
    loaded = [json.loads(l) for l in open(path).readlines()]
    df = pd.json_normalize(loaded)
    df = df[
        [
            col
            for col in df.columns
            if "snapshot" in col or "name" in col or "state" in col
        ]
    ]

    usage = []
    for _, row in df.iterrows():
        start = row.snapshot_start
        for actor_state in [col for col in df.columns if "state" in col]:
            operator_name = re.match(".*Table\((.*)\).*", actor_state).group(1)
            cpu_sum = sum(a["process"]["cpu_percent"] for a in row[actor_state])
            memory_sum = sum(a["process"]["memory_mb"] for a in row[actor_state])
            usage.append(
                {
                    "time": pd.to_datetime(start),
                    "name": operator_name,
                    "cpu": cpu_sum,
                    "memory": memory_sum,
                }
            )
    usage_df = pd.DataFrame(usage)
    return usage_df


def plot_system_metrics(df: pd.DataFrame, plot_save_path: str):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 3))
    for y, ax in zip(["cpu", "memory"], axes):
        sns.lineplot(data=df, x="time", y=y, hue="name", ax=ax)
        ax.set_title(y)
    fig.tight_layout()
    plt.savefig(plot_save_path)


# TODO: fill in model from 1st window if none exists.
def eval_stl_metrics_df(experiment_dir, is_timely_dump=False):
    db_state = {}
    for path in glob(f"{experiment_dir}/client_dump/*"):
        query_timestamp = float(re.match(".*dump/(.*).pkl", path).group(1))
        with open(path, "rb") as f:
            loaded = msgpack.load(f)
        if is_timely_dump:
            timely_state = defaultdict(dict)
            for redis_key, redis_value in loaded.items():
                key = redis_key.split(b"/models")[0].decode()
                if redis_key.endswith(b"/models/value"):
                    py_obj = pickle.loads(redis_value)
                    timely_state[key]["trend"] = py_obj["trend"]
                    timely_state[key]["seasonality"] = py_obj["seasonality"]
                if redis_key.endswith(b"timestamp"):
                    timely_state[key]["timestamp"] = int(redis_value)
            data = dict(timely_state)
        else:  # RALF format
            data = {k.decode(): pickle.loads(v) for k, v in loaded.items()}
        db_state[query_timestamp] = data

    oracle_df_paths = glob(FLAGS.oracle_csv_glob_path)
    key_extractor = re.compile(FLAGS.oracle_csv_extraction_regex)
    yahoo_dfs = {
        int(key_extractor.match(path).group(1)): pd.read_csv(path)
        for path in oracle_df_paths
    }

    scores = []
    for timestamp, db_dump in db_state.items():
        for key, model in db_dump.items():
            seasonal = model["seasonality"]
            model_timestamp = model["timestamp"]
            current_timestamp = int(timestamp * FLAGS.send_rate_per_key)
            prediction = (
                seasonal[(current_timestamp - model_timestamp) % len(seasonal)]
                + model["trend"]
            )
            oracle_df = yahoo_dfs[int(key)]
            if current_timestamp < len(oracle_df):
                oracle_result = oracle_df.iloc[current_timestamp]
                oracle_pred = (
                    oracle_result["pred_trend"] + oracle_result["pred_residual"]
                )
                scores.append(
                    {
                        "timestamp": timestamp,
                        "key": key,
                        "pred": prediction,
                        "oracle_pred": oracle_pred,
                        "event_value": oracle_result["value"],
                        "staleness": current_timestamp - model_timestamp,
                    }
                )
    scores_df = pd.DataFrame(scores)

    losses = []
    for key, ts_df in scores_df.groupby("key"):
        # Can't run msse on time series <= 1 entries.
        if len(ts_df) <= 1:
            loss = None
        else:
            loss = mean_squared_scaled_error(
                y_true=ts_df["oracle_pred"],
                y_pred=ts_df["pred"],
                y_train=ts_df["event_value"],
            )
        losses.append({"key": key, "loss": loss})
    losses_df = pd.DataFrame(losses)

    return scores_df, losses_df


def main(argv):
    analysis_dir = Path(FLAGS.experiment_dir) / "analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    if not FLAGS.is_timely_result:
        system_df = read_system_metrics_df(FLAGS.experiment_dir)
        system_df.to_csv(analysis_dir / "system_df.csv", index=None)
        plot_system_metrics(system_df, analysis_dir / "system_plot.pdf")

    scores_df, losses_df = eval_stl_metrics_df(
        FLAGS.experiment_dir, FLAGS.is_timely_result
    )
    scores_df.to_csv(analysis_dir / "predictions.csv", index=None)
    losses_df.to_csv(analysis_dir / "loss.csv", index=None)

    print("average loss", losses_df["loss"].dropna().mean())
    print(f"Dataframes and plots are in {analysis_dir}")


if __name__ == "__main__":
    app.run(main)
