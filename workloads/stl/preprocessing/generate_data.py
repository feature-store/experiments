from absl import app, flags
import random

import statistics
import pandas as pd
import os
from multiprocessing import Pool
import configparser

from statsmodels.tsa.seasonal import STL


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
    max_seasonality = FLAGS.max_seasonality
    noise = FLAGS.noise 
    data_dir = FLAGS.data_dir

    # TODO: actually extend the time-series 

    source_df = pd.read_csv(os.path.join(data_dir, "yahoo", FLAGS.dataset, f"{source_key}.csv"))

    max_outlier_value, min_outlier_value = max(source_df['noise']), min(source_df['noise'])
    mean, stddev = statistics.mean(source_df['noise']), statistics.stdev(source_df['noise'])
    
    initial_trend = source_df['trend'][0]
    last_trend = source_df['trend'].iloc[-1]
    trend_subtracted_series = source_df['trend'] - initial_trend
    # trend_subtracted_series = np.repeat(trend_subtracted_series, over_sampling_rate)

    seasonality = source_df['seasonality1'] + source_df['seasonality2'] + source_df['seasonality3']
    # seasonality = np.repeat(seasonality, over_sampling_rate)

    repeat_length = (len(trend_subtracted_series) // max_seasonality) * max_seasonality
    print('repeat', repeat_length)

    count = 0
    generated_trend = [last_trend] * max_length
    generated_noise = [0] * max_length
    generated_outlier = [0] * max_length
    generated_seasonality = [0] * max_length

    for i in range(max_length):
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

    new_df = pd.DataFrame({"trend": generated_trend, "noise": generated_noise, "outlier": generated_outlier, "seasonality": generated_seasonality })
    new_df['value'] = new_df['trend'] + new_df['noise'] + new_df['outlier'] + new_df['seasonality']

    assert len(new_df.index) == max_length, f"Wrong length {len(new_df.index)}"


    # assign timestamps 
    interval_ms = int(FLAGS.time_interval_ms * FLAGS.num_keys / FLAGS.num_events)
    timestamp_ms = [ts for ts in range(0, FLAGS.time_interval_ms, interval_ms)]
    new_df["timestamp_ms"] = timestamp_ms

    new_df.to_csv(os.path.join(FLAGS.data_dir, f"extended_{target_key}.csv"))

    return new_df

def fit_stl(keys, data_dir): 
    # calculate baseline STL (ground truth / oracle)
    pass

def fit_stl_window(keys, data_dir, window_size, slide_size=1): 
    # calculate baseline STL (predicted) 
    pass

def create_events_df(keys): 
    df = pd.concat([
        pd.read_csv(
            os.path.join(FLAGS.data_dir, f"extended_{key}.csv")
        ).assign(key_id=key) for key in keys
    ])
    assert len(df.index) == FLAGS.num_events, f"Invalid number of events {len(df.index)}"

    # return sorted by timestamp 
    print("sorting", len(df.index))
    return df.sort_values(by=["timestamp_ms", "key_id"])

def create_queries_df(keys): 

    num_queries_per_key = int(FLAGS.num_queries / len(keys))
    queries_df = pd.DataFrame({
        "key_id": list(keys) * num_queries_per_key, 
        "query_id": range(0, FLAGS.num_queries)
    })

    interval_ms = int(FLAGS.time_interval_ms * FLAGS.num_keys / FLAGS.num_queries) 
    timestamp_ms = [ts for key in keys for ts in range(0, FLAGS.time_interval_ms, interval_ms)]

    queries_df["timestamp_ms"] = timestamp_ms
    return queries_df.sort_values(by=["timestamp_ms"])

def create_features_df(key): 

    df = pd.read_csv(os.path.join(FLAGS.data_dir, f"extended_{key}.csv"))
   
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

    df.to_csv(os.path.join(FLAGS.data_dir, f"best_features_{key}.csv"))
    print("done", os.path.join(FLAGS.data_dir, f"best_features_{key}.csv"))


def create_predictions_df(keys):
    pass

def main(argv):

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

    dataset_name = f"stl-{yahoo_dataset}-{num_keys}-{time_interval_ms}-{num_events}"
    dataset_config = {
        "yahoo_dataset": yahoo_dataset, 
        "num_keys": num_keys, 
        "time_interval_ms": time_interval_ms, 
        "num_events": num_events, 
        "config": config["directory"]
    }

    # read data 
    yahoo_num_keys = 100
    target_keys = range(1, num_keys + 1)
    source_keys = [i % yahoo_num_keys + 1 for i in target_keys]

    max_length = 2000
    noise = 1
    max_seasonality = 7 * 24

    num_workers = 16
    p = Pool(num_workers)
    p.starmap(extend_timeseries, zip(source_keys, target_keys))
    p.close()

    events_df = create_events_df(target_keys)
    events_df.to_csv("events.csv")

    
    queries_df = create_queries_df(target_keys)
    queries_df.to_csv(f"queries.csv")


    p = Pool(num_workers)
    p.map(create_features_df, target_keys)
    p.close()


    # TODO: Log dataset to W&B
    
    #with open(FLAGS.output_path, "w") as f:
    #    json.dump(slide_size_config, f)

    # increase length of time-series 

    # duplicate keys 
    
    # calculate optimal results





if __name__ == "__main__":
    app.run(main)
