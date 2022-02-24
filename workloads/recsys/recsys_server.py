from ast import Global
from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict

import time
import simpy
import os
from absl import app, flags
import ray

# might need to do  export PYTHONPATH='.'
from workloads.util import read_config, use_dataset, log_dataset, log_results, WriteFeatures

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "experiment",
    default=None, 
    help="Experiment name (corresponds to data directory)",
    required=True,
)

flags.DEFINE_string(
    "scheduler",
    default=None, 
    help="Scheduling policy for STL operator", 
    required=True,
)

flags.DEFINE_integer(
    "workers",
    default=1, 
    help="Number of workers for bottleneck operator",
    required=False,
)

#flags.DEFINE_string(
#    "source_file",
#    default=None,
#    help="Dataset file",
#    required=False,
#)
#
#flags.DEFINE_string(
#    "target_dir",
#    default="./results", 
#    help="Result target directory", 
#    required=False,
#)
#
#flags.DEFINE_string(
#    "features_dir",
#    default=None, 
#    help="Directory to read features from", 
#    required=True,
#)

flags.DEFINE_float(
    "learning_rate",
    default=.02,
    help="Learning rate for update step",
    required=False,
)

flags.DEFINE_float(
    "user_feature_reg",
    default=.01,
    help="Regularization term for user features",
    required=False,
)

flags.DEFINE_float(
    "sleep",
    default=.1,
    help="How much to sleep between each timestamp", 
    required=False,
)

@dataclass 
class SourceValue: 
    key: int
    user_id: int
    rating: int
    timestamp: int
    ingest_time: float

@dataclass 
class MovieValue: 
    key: int # movie_id
    movie_features: np.array
    user_id: int
    rating: int
    timestamp: int
    ingest_time: float

@dataclass 
class UserValue: 
    user_id: int # user_id
    user_features: np.array
    timestamp: int
    ingest_time: float
    processing_time: float

@ray.remote
class GlobalTimestamp:
    def __init__(self):
        self.ts = 0
    def incr_ts(self):
        self.ts += 1
    def get_ts(self):
        return self.ts

class DataSource(BaseTransform): 
    #def __init__(self, file_path: str, ts: GlobalTimestamp, sleep: int = 0) -> None:
    def __init__(self, file_path: str, sleep: int = 0) -> None:
        events_df = pd.read_csv(file_path)
        data = dict()
        for timestamp in events_df["timestamp"].unique():
            curr_timestep = events_df[events_df["timestamp"] == timestamp].to_dict('records')
            data[timestamp] = curr_timestep
        self.max_ts = max(list(data.keys()))
        self.ts = 0
        self.data = data
        self.sleep = sleep
        self.last_send_time = -1

    def on_event(self, _: Record) -> List[Record[SourceValue]]:
        #curr_timestamp = await ray.get(self.ts.get_ts.remote())
        curr_timestamp = self.ts
        events = self.data[curr_timestamp]
        #num_remaining = len(self.data[self.data["timestamp"] >= self.ts].index)
        if curr_timestamp == self.max_ts:
            raise StopIteration()
        #self.ts.incr_ts.remote()
        self.ts += 1

        if self.ts % 100 == 0: 
            print("sending", self.ts, self.max_ts)
        time.sleep(self.sleep)
        for e in events: 
            print("sending user", e["user_id"], e["timestamp"])
        return [
            Record(
                SourceValue(
                    key=e["movie_id"], 
                    user_id=e["user_id"], 
                    timestamp=e["timestamp"], 
                    rating=e["rating"],
                    ingest_time=time.time()
                )
            ) for e in events
        ]

class Movie(BaseTransform):
    def __init__(self, movie_features: Dict) -> None:
        self.movie_features = movie_features

    def on_event(self, record: Record):
        movie_id = record.entry.key
        movie_features = self.movie_features[movie_id]
        return Record(MovieValue(key=movie_id, movie_features=movie_features, user_id=record.entry.user_id, rating=record.entry.rating, timestamp=record.entry.timestamp, ingest_time=record.entry.ingest_time))

class User(BaseTransform):
    def __init__(self, user_features: Dict, learning_rate: float, user_feature_reg: float) -> None:
        self.user_features = user_features
        self.learning_rate = learning_rate
        self.user_feature_reg = user_feature_reg

    def on_event(self, record: Record):
        user_id = record.entry.user_id
        user_features = self.user_features[user_id]
        movie_features = record.entry.movie_features
        
        prediction = user_features.dot(movie_features.T)
        error = record.entry.rating - prediction
        updated_user_features = user_features + self.learning_rate * (error * movie_features - self.user_feature_reg * user_features)
        self.user_features[user_id] = updated_user_features

        print("Updating user", user_id, record.entry.timestamp)

        return Record(UserValue(user_id=user_id, user_features=updated_user_features.tolist(), timestamp=record.entry.timestamp, ingest_time=record.entry.ingest_time, processing_time=time.time()))

#class WriteFeatures(BaseTransform): 
#    def __init__(self, file_path: str, timestamp: GlobalTimestamp):
#        df = pd.DataFrame({"user_id": [], "user_features": [], "ingest_timestamp": [], "timestamp": []})
#        self.filename = file_path 
#        self.ts = timestamp
#        print("WRITING TO", self.filename)
#        df.to_csv(self.filename, index=None)
#
#    def on_event(self, record: Record): 
#        curr_timestamp = ray.get(self.ts.get_ts.remote())
#        df = pd.DataFrame({'user_id': [record.entry.key], 'ingest_timestamp': [record.entry.timestamp], 'user_features': [list(record.entry.user_features)], 'timestamp': [curr_timestamp]})
#        temp_csv = df.to_csv(index=None, header=None)
#        #record_csv = f"{record.entry.key}, {list(record.entry.user_features)}, {curr_timestamp}\n"
#        #print(record_csv == temp_csv)
#        with open(self.filename, "a") as file:
#            file.write(temp_csv)
#        #print("wrote", record.entry.key, record.entry.timestamp)

def get_features(file_path):
    df = pd.read_csv(file_path)
    features = dict()
    for row in df.itertuples():
        features[row.id] = np.array(eval(row.features))
    return features 

def main(argv):
    print("Running Recsys pipeline on ralf...")

    data_dir = use_dataset(FLAGS.experiment, redownload=False)
    results_dir = os.path.join(read_config()["results_dir"], FLAGS.experiment)
    name = f"results_workers_{FLAGS.workers}_{FLAGS.scheduler}_learningrate_{FLAGS.learning_rate}_userfeaturereg_{FLAGS.user_feature_reg}_sleep_{FLAGS.sleep}"
    print("dataset", data_dir)

    ## create results file/directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/{name}.csv"
    print("results file", results_file)

    # read data
    user_features = get_features(f"{data_dir}/user_features.csv")
    movie_features = get_features(f"{data_dir}/movie_features.csv")
    ratings_file = f"{data_dir}/ratings.csv"

    #deploy_mode = "ray"
    deploy_mode = "ray"
    #deploy_mode = "simpy"
    app = RalfApplication(RalfConfig(deploy_mode=deploy_mode))

    # create simulation env 
    if deploy_mode == "simpy": 
        env = simpy.Environment()
    else: 
        env = None

    schedulers = {
        "fifo": FIFO(), 
        "lifo": LIFO(), 
    }

    # create feature frames
    # TODO: benchmark to figure out better processing_time values for simulation
    timestamp = GlobalTimestamp.remote()
    movie_ff = app.source(
        #DataSource(f"{data_dir}/ratings.csv", timestamp, FLAGS.sleep),
        DataSource(f"{data_dir}/ratings.csv", FLAGS.sleep),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
                stop_after_s=10
            ),         
            ray_config=RayOperatorConfig(num_replicas=1)
    )
    ).transform(
        Movie(movie_features), 
        scheduler=FIFO(), 
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=1)
        )
    )
    user_ff = movie_ff.transform(
        User(user_features, FLAGS.learning_rate, FLAGS.user_feature_reg),
        scheduler=schedulers[FLAGS.scheduler],
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.2, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers)
        )
    ).transform(
        WriteFeatures(results_file, ["user_id", "user_features", "ingest_time", "timestamp", "processing_time"])
    )

    app.deploy()

    if deploy_mode == "simpy": env.run(100)
    app.wait()

    #print("logging to wandb")
    #if FLAGS.source_file:
    #    run = wandb.init(project="ralf-stl", entity="ucb-ralf")
    #target_artifact = wandb.Artifact(f"{FLAGS.experiment}-results", type='results')
    #target_artifact.add_dir(results_dir)
    #run.log_artifact(target_artifact)
    print("Completed run!")
    log_results(name)


if __name__ == "__main__":
    app.run(main)

