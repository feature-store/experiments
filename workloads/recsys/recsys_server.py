from ast import Global
from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record, BaseScheduler
from ralf.v2.scheduler import LeastUpdate
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
import time

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


class LeastUpdated(BaseScheduler):
    def __init__(self, size: int) -> None:
        self.waker: Optional[threading.Event] = None
        self.stop_iteration = None
        self.size = size

        #self.queue = {key: [] for key in keys}
        #self.keys = keys
        self.queue = {}

    def push_event(self, record: Record):
        self.wake_waiter_if_needed()
        if record.is_stop_iteration(): # stop iteration
            print("GOT STOP")
            self.stop_iteration = record

        if record.user_id in self.queue: 
            self.queue[record.user_id].append(record)
        else: 
            self.queue[record.user_id] = [record]
            
    def pop_event(self) -> Record:
        if self.stop_iteration: # return stop iteration record
            print("POP STOP")
            return self.stop_iteration
        if len(self.queue) == 0:
            return Record.make_wait_event(self.new_waker())

        return self.queue.pop(0)


class FIFOSize(BaseScheduler):
    def __init__(self, size: int) -> None:
        self.waker: Optional[threading.Event] = None
        self.stop_iteration = None
        self.size = size

        self.queue: List[Record] = []

    def push_event(self, record: Record):
        self.wake_waiter_if_needed()
        self.queue.append(record)
        if record.is_stop_iteration(): # stop iteration
            print("GOT STOP")
            self.stop_iteration = record
        if len(self.queue) > self.size: 
            print(f"Queue too large {self.size}, {len(self.queue)}")
            self.queue = self.queue[len(self.queue) - int(self.size/2):]
            
    def pop_event(self) -> Record:
        print(len(self.queue))
        if self.stop_iteration: # return stop iteration record
            print("POP STOP")
            return self.stop_iteration
        if len(self.queue) == 0:
            return Record.make_wait_event(self.new_waker())

        return self.queue.pop(0)




@dataclass 
class SourceValue: 
    movie_id: int
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
    runtime: float

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
        return [
            Record(
                SourceValue(
                    movie_id=e["movie_id"], 
                    user_id=e["user_id"], 
                    timestamp=e["timestamp"], 
                    rating=e["rating"],
                    ingest_time=time.time()
                )
            ) for e in events
        ]

#class Movie(BaseTransform):
#    def __init__(self, movie_features: Dict) -> None:
#        self.movie_features = movie_features
#
#    def on_event(self, record: Record):
#        movie_id = record.entry.key
#        movie_features = self.movie_features[movie_id]
#        return Record(MovieValue(key=movie_id, movie_features=movie_features, user_id=record.entry.user_id, rating=record.entry.rating, timestamp=record.entry.timestamp, ingest_time=record.entry.ingest_time))

class User(BaseTransform):
    def __init__(self, movie_features: Dict, user_features: Dict, learning_rate: float, user_feature_reg: float) -> None:
        self.movie_features = movie_features
        self.user_features = user_features
        self.learning_rate = learning_rate
        self.user_feature_reg = user_feature_reg
        self.num_updates = 0

    def on_event(self, record: Record):
        user_id = record.entry.user_id
        movie_id = record.entry.movie_id
        user_features = self.user_features[user_id]
        movie_features = self.movie_features[movie_id]
        #movie_features = record.entry.movie_features
       
        st = time.time()


        Users[i] = np.linalg.solve(
                np.dot(movie_matrix, np.dot(np.diag(Ri), movie_matrix.T)) + self.user_feature_reg * np.eye(len(user_features)),
                np.dot(Items, np.dot(np.diag(Ri), A[i].T))
        ).T
        prediction = user_features.dot(movie_features.T)
        error = record.entry.rating - prediction
        updated_user_features = user_features + self.learning_rate * (error * movie_features - self.user_feature_reg * user_features)
        self.user_features[user_id] = updated_user_features

        #print("Updating user", user_id, record.entry.timestamp)
        runtime = time.time() - st

        if self.num_updates % 100 == 0:
            print("Num updates", self.num_updates)
        self.num_updates += 1

        return Record(UserValue(user_id=user_id, user_features=updated_user_features.tolist(), timestamp=record.entry.timestamp, ingest_time=record.entry.ingest_time, processing_time=time.time(), runtime=runtime))

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
        "fifo-100": FIFOSize(100), 
        "fifo-1000": FIFOSize(1000), 
        "lifo": LIFO(), 
        "least": LeastUpdate(),
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
    #).transform(
    #    Movie(movie_features), 
    #    scheduler=FIFO(), 
    #    operator_config=OperatorConfig(
    #        simpy_config=SimpyOperatorConfig(
    #            shared_env=env, 
    #            processing_time_s=0.01, 
    #        ),         
    #        ray_config=RayOperatorConfig(num_replicas=FLAGS.workers)
    #    )
    #)
    #user_ff = movie_ff.transform(
    ).transform(
        User(movie_features, user_features, FLAGS.learning_rate, FLAGS.user_feature_reg),
        scheduler=schedulers[FLAGS.scheduler],
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.2, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers)
        )
    ).transform(
        WriteFeatures(results_file, ["user_id", "user_features", "ingest_time", "timestamp", "processing_time", "runtime"])
    )

    app.deploy()

    if deploy_mode == "simpy": env.run(100)
    app.wait()

    print("Completed run!")
    log_results(name)


if __name__ == "__main__":
    app.run(main)
##
