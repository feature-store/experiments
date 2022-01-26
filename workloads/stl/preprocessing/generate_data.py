from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dataset",
    default="A4",
    help="Yahoo dataset name"
)

flags.DEFINE_integer(
    "num_keys",
    default=None,
    help="Total number of keys in dataset"
    required=True,
)

flags.DEFINE_integer(
    "num_keys",
    default=None,
    help="Total number of keys in dataset"
    required=True,
)

flags.DEFINE_integer(
    "time_interval_ms",
    default=None,
    help="Total time interval" 
    required=True,
)

# TODO: potentially implent option for rate of updates
flags.DEFINE_integer(
    "num_events"
    default=None,
    help="Total number of events"
    required=True,
)

def extend_timeseries(source_key, target_key, data_dir, max_length, noise, max_seasonality):
    # write key CSV
    # from https://github.com/feature-store/experiments/blob/simon-stl/stl/extend_yahoo_dataset.ipynb
    return data 

def fit_stl(keys, data_dir): 
    # calculate baseline STL (ground truth)
    pass

def fit_stl_window(keys, data_dir, window_size, slide_size=1): 
    # calculate baseline STL (predicted)
    pass

def join_key_dfs(keys, data_dir): 
    pass

def create_events_df(keys, data_dir, time_interval_ms):

    events_df = join_key_dfs(keys, data_dir)
    assert len(events_df.index) == num_events

    # assign timestamps 
    interval_ms = int(time_interval_ms / num_events)
    timestamps_ms = [ts for key in keys for ts in range(0, time_interval_ms, interval_ms)]

    # write csv
    events_df["timestamp_ms"] = timestamp_ms
    events.to_csv(f"{data_dir}/events.csv")

    return events_df

def create_queries_df(keys, num_queries, data_dir, time_interval_ms): 

    num_queries_per_key = int(num_queries / len(keys))
    queries_df = pd.DataFrame({
        "keys": keys * num_queries_per_key, 
        "query_id": range(0, num_queries)
    })

    interval_ms = int(time_interval_ms / num_queries) 
    timestamps_ms = [ts for key in keys for ts in range(0, time_interval_ms, interval_ms)]

    queries_df["timestamp_ms"] = timestamp_ms
    queries_df.to_csv(f"{data_dir}/queries.csv")

def create_features_df(keys, window_size, slide_size): 
    pass

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

    dataset_name = f"stl-{yahoo_dataset)-{num_keys}-{time_interval_ms}-{num_events}"
    dataset_config = {
        "yahoo_dataset": yahoo_dataset, 
        "num_keys": num_keys, 
        "time_interval_ms": time_interval_ms, 
        "num_events": num_events, 
        "config": config["directory"]
    }

    # read data 
    yahoo_num_keys = 100
    target_keys = range(1, num_keys)
    source_keys = [i % yahoo_num_keys + 1 for i in range(1, num_keys)]


    num_workers = 32
    p = Pool(num_workers)
    p.map(extend_timeseries, [(src_key, trg_key, data_dir, max_length, noise, max_seasonality) for src_key, trg_key in zip(source_keys, target_keys)])

    

    # increase length of time-series 
    extended_points = extend_time_series(data



    # TODO: Log dataset to W&B
    
    with open(FLAGS.output_path, "w") as f:
        json.dump(slide_size_config, f)

    # increase length of time-series 

    # duplicate keys 
    
    # calculate optimal results





if __name__ == "__main__":
    app.run(main)
