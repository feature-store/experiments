from absl import app, flags
import random
import json

import statistics
import pandas as pd
import os
from multiprocessing import Pool
import configparser
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL

import sys 
sys.path.insert(1, "../")
from util import upload_dataset

import wandb


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir",
    default=None,
    help="Dataset directory",
    required=True,
)


flags.DEFINE_string(
    "dataset",
    default="A4",
    help="Yahoo dataset name",
)

flags.DEFINE_integer(
    "num_keys",
    default=None,
    help="Total number of keys in dataset",
    required=True,
)

flags.DEFINE_integer(
    "time_interval_ms",
    default=None,
    help="Total time interval",
    required=True,
)

flags.DEFINE_integer(
    "num_queries",
    default=None,
    help="Total number of queries", 
    required=True
)
# TODO: potentially implent option for rate of updates
flags.DEFINE_integer(
    "num_events",
    default=None,
    help="Number of update events", 
    required=True
)

flags.DEFINE_integer(
    "seasonality",
    default=12,
    help="Max seasonality for generated time-series"
)

flags.DEFINE_integer(
    "window_size",
    default=24,
    help="Max seasonality for generated time-series"
)

# configuration for generating extended time-series
flags.DEFINE_integer(
    "max_seasonality",
    default=168,
    help="Max seasonality for generated time-series"
)

flags.DEFINE_integer(
    "noise",
    default=100,
    help="Noise to add to generated time-series"
)

def extend_timeseries(source_key, target_key):
    # write key CSV
    # from https://github.com/feature-store/experiments/blob/simon-stl/stl/extend_yahoo_dataset.ipynb

    max_length = int(FLAGS.num_events / FLAGS.num_keys) # number events per key
    print("extending to", max_length)
    max_seasonality = FLAGS.max_seasonality
    noise = FLAGS.noise 

    # TODO: actually extend the time-series 

    source_df = pd.read_csv(os.path.join(FLAGS.data_dir, "yahoo", FLAGS.dataset, f"{source_key}.csv"))

    max_outlier_value, min_outlier_value = max(source_df['noise']), min(source_df['noise'])
    mean, stddev = statistics.mean(source_df['noise']), statistics.stdev(source_df['noise'])
    
    initial_trend = source_df['trend'][0]
    last_trend = source_df['trend'].iloc[-1]
    trend_subtracted_series = source_df['trend'] - initial_trend
    # trend_subtracted_series = np.repeat(trend_subtracted_series, over_sampling_rate)

    seasonality = source_df['seasonality1'] + source_df['seasonality2'] + source_df['seasonality3']
    # seasonality = np.repeat(seasonality, over_sampling_rate)

    repeat_length = (len(trend_subtracted_series) // max_seasonality) * max_seasonality
    #print('repeat', repeat_length)

    count = 0
    generated_length = max_length - len(source_df.index) 
    generated_trend = [last_trend] * generated_length
    generated_noise = [0] * generated_length
    generated_outlier = [0] * generated_length
    generated_seasonality = [0] * generated_length

    #print(repeat_length, len(trend_subtracted_series))

    for i in range(generated_length):
        if count >= repeat_length:
            count = 0
            last_trend = generated_trend[i-1]

        generated_trend[i] = last_trend + trend_subtracted_series[count]
        generated_seasonality[i] = seasonality[count]
        generated_noise[i] = random.gauss(mean, stddev)
        generated_outlier[i] = 0
        if random.randint(0, 100) > 100 - noise:
            if random.randint(0, 100) > 50:
                generated_outlier[i] = max_outlier_value * random.randint(70,100) // 100
            else:
                generated_outlier[i] = min_outlier_value * random.randint(70,100) // 100
        count += 1

    #print(seasonality)
    #print(len(generated_seasonality))
    new_df = pd.DataFrame({
        "trend": source_df.trend.tolist() + generated_trend, 
        "noise": source_df.noise.tolist() + generated_noise, 
        "outlier": source_df.anomaly.tolist() + generated_outlier, 
        "seasonality": seasonality.tolist() + generated_seasonality, 
    })
    new_df['value'] = new_df['trend'] + new_df['noise'] + new_df['outlier'] + new_df['seasonality']

    assert len(new_df.index) == max_length, f"Wrong length {len(new_df.index)}"


    # assign timestamps 
    interval_ms = int(FLAGS.time_interval_ms * FLAGS.num_keys / FLAGS.num_events)
    timestamp_ms = [ts for ts in range(0, FLAGS.time_interval_ms, interval_ms)]
    #print(timestamp_ms)
    #print(len(timestamp_ms), len(new_df.index))
    new_df["timestamp_ms"] = timestamp_ms
    #print(data_dir)
    #new_df.to_csv(os.path.join(data_dir, "extended_data", f"{target_key}.csv"))
    #source_df.to_csv(os.path.join(data_dir, "data", f"{target_key}.csv"))

    return new_df, source_df

def fit_stl(keys, data_dir): 
    # calculate baseline STL (ground truth / oracle)
    pass

def fit_stl_window(keys, data_dir, window_size, slide_size=1): 
    # calculate baseline STL (predicted) 
    pass

def create_events_df(keys, prefix=""): 
    df = pd.concat([
        pd.read_csv(
            f"{prefix}{key}.csv"
        ).assign(key_id=key) for key in keys
    ])
    assert len(df.index) == FLAGS.num_events, f"Invalid number of events {len(df.index)}"

    # return sorted by timestamp 
    print("sorting", len(df.index))
    return df.sort_values(by=["timestamp_ms", "key_id"])

def create_queries_df(keys, data_dir, start_index=0): 

    num_queries_per_key = int(FLAGS.num_queries / len(keys))

    key_col = []
    val_col = []
    time_col = []
    for key in keys: 
        df = pd.read_csv(os.path.join(data_dir, f"{key}.csv"))
        interval = int(len(df.index) / num_queries_per_key)
        v = df[df.index % interval == 0].value.tolist()
        t = df[df.index % interval == 0].timestamp_ms.tolist()
        assert len(v) == num_queries_per_key, f"Invalid length {len(v)}, {num_queries_per_key}, {len(df.index)}, interval {interval}"
        assert len(t) == num_queries_per_key 

        key_col += [key] * num_queries_per_key
        val_col += v
        time_col += t


    print(len(key_col), FLAGS.num_queries)

    queries_df = pd.DataFrame({
        "key_id": key_col, 
        "value": val_col, 
        "timestamp_ms": time_col,
        "query_id": range(0, FLAGS.num_queries)
    })

    #queries_df = pd.DataFrame({ "key_id": list(keys) * num_queries_per_key, "query_id": range(0, FLAGS.num_queries)
    #})

    #interval_ms = int(FLAGS.time_interval_ms * FLAGS.num_keys / FLAGS.num_queries) 
    #timestamp_ms = [ts for key in keys for ts in range(0, FLAGS.time_interval_ms, interval_ms)]

    #queries_df["timestamp_ms"] = timestamp_ms
    return queries_df.sort_values(by=["timestamp_ms"])

def create_features_df(key, data_dir): 

    df = pd.read_csv(os.path.join(data_dir, "extended_data", f"{key}.csv"))
   
    seasonality = []
    trend = []

    windows = []
    for window in df.value.rolling(window=FLAGS.window_size): 
        if len(window) < FLAGS.window_size:
            seasonality.append(None)
            trend.append(None)
        else:
            stl = STL(window.tolist(), period=FLAGS.seasonality, robust=True).fit()
            trend.append(stl.trend[-1])
            seasonality.append(stl.seasonal[-(FLAGS.seasonality + 1) : -1])

    assert len(seasonality) == len(df.index)
    assert len(trend) == len(df.index)

    df["seasonality"] = seasonality
    df["trend"] = trend 
    df["key_id"] = key

    #df.to_csv(os.path.join(data_dir, "oracle", f"features_{key}.csv"))
    print("done", f"features_{key}.csv")
    return df


def create_predictions_df(keys):
    pass

def main(argv):

    run = wandb.init(project="ralf-stl", entity="ucb-ralf", job_type="dataset-creation")

    # dataset configuration
    yahoo_dataset = FLAGS.dataset
    num_keys = FLAGS.num_keys
    time_interval_ms = FLAGS.time_interval_ms
    num_events = FLAGS.num_events

    # configuration file
    config = configparser.ConfigParser()
    config.read("config.yml") 
    results_dir = config["directory"]["results_dir"]
    data_dir = os.path.join(config["directory"]["data_dir"], yahoo_dataset)

    dataset_name = f"stl-{yahoo_dataset}-keys-{num_keys}-interval-{time_interval_ms}-events-{num_events}"
    print(dataset_name)
    dataset_config = {
        "yahoo_dataset": yahoo_dataset, 
        "num_keys": num_keys, 
        "time_interval_ms": time_interval_ms, 
        "num_events": num_events, 
        "config": dict(config["directory"])
    }
    
    # setup experiment directory
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)
        os.mkdir(f"{dataset_name}/extended_data")
        os.mkdir(f"{dataset_name}/data")
        os.mkdir(f"{dataset_name}/oracle")

    open(f"{dataset_name}/config.json", "w").write(json.dumps(dataset_config))


    # read data 
    yahoo_num_keys = 100
    target_keys = range(1, num_keys + 1)
    source_keys = [i % yahoo_num_keys + 1 for i in target_keys]

    num_workers = 8
    p = Pool(num_workers)
    dfs = p.starmap(extend_timeseries, zip(source_keys, target_keys))
    p.close()

    for df, key in tqdm(zip(dfs, target_keys)): 
        df[0].to_csv(f"{dataset_name}/extended_data/{key}.csv")
        df[1].to_csv(f"{dataset_name}/data/{key}.csv")

    #events_df = create_events_df(target_keys, prefix="yahoo/A4/")
    events_df = create_events_df(target_keys, prefix=f"{dataset_name}/extended_data/")
    events_df.to_csv(f"{dataset_name}/events.csv")

    assert events_df.timestamp_ms.max() < FLAGS.time_interval_ms
    
    queries_df = create_queries_df(target_keys, data_dir=f"{dataset_name}/extended_data")
    queries_df.to_csv(f"{dataset_name}/queries.csv")

    assert queries_df.timestamp_ms.max() < FLAGS.time_interval_ms

    p = Pool(num_workers)
    dfs = p.starmap(create_features_df, [(key, dataset_name) for key in target_keys])
    p.close()

    for df, key in tqdm(zip(dfs, target_keys)): 
        df.to_csv(f"{dataset_name}/oracle/{key}.csv")

    oracle_df = pd.concat(dfs).set_index(["key_id", "timestamp_ms"], drop=False)
    oracle_df.to_csv(f"{dataset_name}/oracle_features.csv")


    #df.to_csv(os.path.join(data_dir, "oracle", f"features_{key}.csv"))


    # TODO: Log dataset to W&B
    
    #with open(FLAGS.output_path, "w") as f:
    #    json.dump(slide_size_config, f)

    # increase length of time-series 

    # duplicate keys 
    
    # calculate optimal results
    #upload_dataset(dataset_name, os.path.basename(dataset_name))
    artifact = wandb.Artifact(dataset_name, type="dataset")
    artifact.add_dir(dataset_name)
    run.log_artifact(artifact)





if __name__ == "__main__":
    app.run(main)
