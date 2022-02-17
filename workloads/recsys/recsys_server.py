from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict

import simpy
import os
from absl import app, flags
import wandb

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "experiment",
    default=None, 
    help="Experiment name",
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
    default=2, 
    help="Number of workers for bottleneck operator",
    required=False,
)

flags.DEFINE_string(
    "source_file",
    default=None,
    help="Dataset file",
    required=False,
)

flags.DEFINE_string(
    "target_dir",
    default="./results", 
    help="Result target directory", 
    required=False,
)

flags.DEFINE_string(
    "features_dir",
    default=None, 
    help="Directory to read features from", 
    required=True,
)

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

@dataclass 
class SourceValue: 
    key: int
    user_id: int
    rating: int
    timestamp: int

@dataclass 
class MovieValue: 
    key: int # movie_id
    movie_features: np.array
    user_id: int
    rating: int
    timestamp: int

@dataclass 
class UserValue: 
    key: int # user_id
    user_features: np.array
    timestamp: int

class DataSource(BaseTransform): 
    def __init__(self, file_path: str) -> None:
        events_df = pd.read_csv(file_path)

        self.ts = 0
        self.data = events_df
        self.last_send_time = -1

    def on_event(self, _: Record) -> List[Record[SourceValue]]:
    
        # TODO: fix this - very slow
        events = self.data[self.data["timestamp"] == self.ts].to_dict('records')
        num_remaining = len(self.data[self.data["timestamp"] >= self.ts].index)
        if num_remaining == 0:
            raise StopIteration()
        if len(events) > 0:
            print("sending events", self.ts, len(events), "remaining", num_remaining)
        self.ts += 1
        return [
            Record(
                SourceValue(key=e["movie_id"], user_id=e["user_id"], timestamp=e["timestamp"], rating=e["rating"])
            ) for e in events
        ]

class Movie(BaseTransform):
    def __init__(self, movie_features: Dict) -> None:
        self.movie_features = movie_features

    def on_event(self, record: Record):
        movie_id = record.entry.key
        movie_features = self.movie_features[movie_id]
        return Record(MovieValue(key=movie_id, movie_features=movie_features, user_id=record.entry.user_id, rating=record.entry.rating, timestamp=record.entry.timestamp))

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

        return Record(UserValue(key=user_id, user_features=updated_user_features, timestamp=record.entry.timestamp))

class WriteFeatures(BaseTransform): 
    def __init__(self, file_path: str):
        df = pd.DataFrame({"user_id": [], "user_features": [], "timestamp": []})
        self.filename = file_path 
        df.to_csv(self.filename, index=None)

    def on_event(self, record: Record): 
        df = pd.DataFrame({'user_id': [record.entry.key], 'user_features': [list(record.entry.user_features)], 'timestamp': [record.entry.timestamp]})
        temp_csv = df.to_csv(index=None, header=None)
        with open(self.filename, "a") as file:
            file.write(temp_csv)
        print("wrote", record.entry.key, record.entry.timestamp)

def get_features(file_path):
    df = pd.read_csv(file_path)
    features = dict()
    for row in df.itertuples():
        features[row.id] = np.array(eval(row.features))
    return features 

def main(argv):
    print("Running Recsys pipeline on ralf...")


    if FLAGS.source_file is None:
        # download data 
        run = wandb.init(project="ralf-recsys", entity="ucb-ralf")
        src_artifact = run.use_artifact(f"{FLAGS.experiment}:latest", type='dataset')
        data_dir = src_artifact.download()
        print(data_dir)
    else: 
        # use existing data dir
        source_file = FLAGS.source_file

    # create results file/directory
    results_dir = f"{FLAGS.target_dir}/{FLAGS.experiment}"
    if not os.path.isdir(results_dir): 
        os.mkdir(results_dir)
    results_file = f"{results_dir}/results_workers_{FLAGS.workers}_{FLAGS.scheduler}_learningrate_{FLAGS.learning_rate}_userfeaturereg_{FLAGS.user_feature_reg}.csv"

    features_dir = FLAGS.features_dir

    user_features = get_features(f"{features_dir}/user_features.csv")
    movie_features = get_features(f"{features_dir}/movie_features.csv")

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
    movie_ff = app.source(
        DataSource(source_file),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
                stop_after_s=10
            ),         
            ray_config=RayOperatorConfig(num_replicas=2)
    )
    ).transform(
        Movie(movie_features), 
        scheduler=FIFO(), 
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=2)
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
        WriteFeatures(results_file)
    )

    app.deploy()

    if deploy_mode == "simpy": env.run(100)
    app.wait()

    print("logging to wandb")
    if FLAGS.source_file:
        run = wandb.init(project="ralf-stl", entity="ucb-ralf")
    target_artifact = wandb.Artifact(f"{FLAGS.experiment}-results", type='results')
    target_artifact.add_dir(results_dir)
    run.log_artifact(target_artifact)
    print("Completed run!")


if __name__ == "__main__":
    app.run(main)

