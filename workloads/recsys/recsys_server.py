from ast import Global
import json
import pickle
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
from workloads.recsys.als import runALS

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

flags.DEFINE_string(
    "update",
    default=None, 
    help="Update type", 
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
        print(events_df.columns)
        events_df = events_df.iloc[: , 1:]
        print(events_df.columns)
        events_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        data = dict()
        for timestamp in events_df["timestamp"].unique():
            curr_timestep = events_df[events_df["timestamp"] == timestamp].to_dict('records')
            data[timestamp] = curr_timestep
        self.max_ts = max(list(data.keys()))
        self.ts = events_df["timestamp"].min()
        self.data = data
        self.sleep = sleep
        self.last_send_time = -1

    def on_event(self, _: Record) -> List[Record[SourceValue]]:
        #curr_timestamp = await ray.get(self.ts.get_ts.remote())
        curr_timestamp = self.ts
        if self.ts in self.data: 
            events = self.data[self.ts]
        else: 
            events = []
        
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
                entry=SourceValue(
                    movie_id=e["movie_id"], 
                    user_id=e["user_id"], 
                    timestamp=e["timestamp"], 
                    rating=e["rating"],
                    ingest_time=time.time()
                ),
                shard_key=str(e["user_id"])
            ) for e in events
        ]

class UserSGD(BaseTransform):
    """
    Maintain user embeddings with SGD updates
    """

    def __init__(self, data_dir: str, learning_rate: float, user_feature_reg: float) -> None:
        print(f"{data_dir}/movie_to_index.json")
        self.movie_to_index = json.load(open(f"{data_dir}/movie_to_index.json", "r"))
        self.user_to_index = json.load(open(f"{data_dir}/user_to_index.json", "r"))
        self.user_matrix = pickle.load(open(f"{data_dir}/user_matrix.pkl", "rb"))
        self.movie_matrix = pickle.load(open(f"{data_dir}/movie_matrix.pkl", "rb"))

        self.lr = learning_rate
        self.reg = user_feature_reg
        self.num_updates = 0

    def on_event(self, record: Record):
        ui = self.user_to_index[record.entry.user_id]
        mi = self.movie_to_index[record.entry.movie_id]
        rating = record.entry.rating

        st = time.time()
        user_features = self.user_matrix[ui]
        prediction = user_features.dot(self.movie_matrix[mi].T)
        error = rating - prediction
        
        # update user matrix
        self.user_matrix[ui] = user_features + self.lr * (error * self.movie_matrix[mi] - self.reg * user_features)
        runtime = time.time() - st

        if self.num_updates % 100 == 0:
            print("Num updates", self.num_updates)
        self.num_updates += 1

        return Record(
            entry=UserValue(
                user_id=user_id, 
                user_features=self.user_matrix[ui].tolist(), 
                timestamp=record.entry.timestamp, 
                ingest_time=record.entry.ingest_time, 
                processing_time=time.time(), 
                runtime=runtime
            ), 
            shard_key=str(user_id)
        )

class UserALSRow(BaseTransform): 
    """
    Maintain user embeddings by re-solving for user vector
    """
    def __init__(self, data_dir: str, user_feature_reg: int = 10):
        self.movie_to_index = json.load(open(f"{data_dir}/movie_to_index.json", "r"))
        self.user_to_index = json.load(open(f"{data_dir}/user_to_index.json", "r"))
        self.A = pickle.load(open(f"{data_dir}/A.pkl", "rb")).tolist()
        self.R = pickle.load(open(f"{data_dir}/R.pkl", "rb")).tolist()
        self.user_matrix = pickle.load(open(f"{data_dir}/user_matrix.pkl", "rb")).tolist()
        self.movie_matrix = pickle.load(open(f"{data_dir}/movie_matrix.pkl", "rb")).tolist()
        self.num_updates = 0 
        self.user_feature_reg = user_feature_reg

    def on_event(self, record: Record):
        ui = self.user_to_index[str(record.entry.user_id)]
        mi = self.movie_to_index[str(record.entry.movie_id)]
        st = time.time()

        # required to make things writable... idk why
        self.A = np.array(self.A)
        self.R = np.array(self.R)
        self.user_matrix = np.array(self.user_matrix)
        self.movie_matrix = np.array(self.movie_matrix)

        # update matrix
        self.A[ui][mi] = record.entry.rating
        self.R[ui][mi] = 1

        Ri = self.R[ui] 
        user_features = self.user_matrix[ui]
        n_factors = len(user_features)

        # calculate new user vector
        self.user_matrix[ui] = np.linalg.solve(
            np.dot(self.movie_matrix.T, np.dot(np.diag(Ri), self.movie_matrix)) + self.user_feature_reg * np.eye(n_factors),
            np.dot(self.movie_matrix.T, np.dot(np.diag(Ri), self.A[ui].T))
        ).T

        runtime = time.time() - st
        print(f"Updated {ui}, {mi}, runtime: {runtime}")

        if self.num_updates % 100 == 0:
            print("Num updates", self.num_updates)
        self.num_updates += 1

        return Record(
            entry=UserValue(
                user_id=record.entry.user_id, 
                user_features=self.user_matrix[ui].tolist(), 
                timestamp=record.entry.timestamp, 
                ingest_time=record.entry.ingest_time, 
                processing_time=time.time(), 
                runtime=runtime
            ),
            shard_key=str(record.entry.user_id)
        )

class UserALS(BaseTransform): 
    """
    Maintain user embeddings by re-solving both user/movie matrix
    """
    def __init__(self, data_dir: str, user_feature_reg: int = 0.1, n_iter: int = 2):
        self.movie_to_index = json.load(open(f"{data_dir}/movie_to_index.json", "r"))
        self.user_to_index = json.load(open(f"{data_dir}/user_to_index.json", "r"))
        self.A = pickle.load(open(f"{data_dir}/A.pkl", "rb")).tolist()
        self.R = pickle.load(open(f"{data_dir}/R.pkl", "rb")).tolist()
        self.user_matrix = pickle.load(open(f"{data_dir}/user_matrix.pkl", "rb")).tolist()
        self.movie_matrix = pickle.load(open(f"{data_dir}/movie_matrix.pkl", "rb")).tolist()
        self.num_updates = 0 

    def on_event(self, record: Record):
        ui = self.user_to_index[str(record.entry.user_id)]
        mi = self.movie_to_index[str(record.entry.movie_id)]

        # required to make things writable... idk why
        self.A = np.array(self.A)
        self.R = np.array(self.R)
        self.user_matrix = np.array(self.user_matrix)
        self.movie_matrix = np.array(self.movie_matrix)

        st = time.time()

        # update matrix
        self.A[ui][mi] = record.entry.rating
        self.R[ui][mi] = 1

        Ri = self.R[ui] 
        user_features = user_matrix[ui]
        n_factors = len(user_features)
        self.user_matrix, self.movie_matrix = runALS(
    	    A_matrix_batch, 
    	    R_matrix_batch, 
    	    n_factors, 
    	    n_iter, 
    	    reg, 
    	    streaming_user_matrix_batch, 
    	    streaming_movie_matrix_batch, 
    	    users=None, # [ui], # filter user
            gpu=False, 
    	)

        runtime = time.time() - st

        print(f"Updated {ui}, {mi}, runtime: {runtime}")

        if self.num_updates % 100 == 0:
            print("Num updates", self.num_updates)
        self.num_updates += 1

        return Record(
            entry=UserValue(
                user_id=record.entry.user_id, 
                user_features=self.user_matrix[ui].tolist(), 
                timestamp=record.entry.timestamp, 
                ingest_time=record.entry.ingest_time, 
                processing_time=time.time(), 
                runtime=runtime
            ), 
            shard_key=str(record.entry.user_id)
        )

def main(argv):
    print("Running Recsys pipeline on ralf...")

    data_dir = use_dataset(FLAGS.experiment, redownload=False)
    results_dir = os.path.join(read_config()["results_dir"], FLAGS.experiment)
    name = f"results_workers_{FLAGS.update}_{FLAGS.workers}_{FLAGS.scheduler}_learningrate_{FLAGS.learning_rate}_userfeaturereg_{FLAGS.user_feature_reg}_sleep_{FLAGS.sleep}"
    print("dataset", data_dir)

    ## create results file/directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/{name}.csv"
    print("results file", results_file)

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
        "fifo": FIFO(), 
        "least": LeastUpdate(),
    }

    operators = {
        "sgd": UserSGD(data_dir, FLAGS.learning_rate, FLAGS.user_feature_reg),
        "user": UserALSRow(data_dir), 
        "als": UserALS(data_dir), 
    }

    # create feature frames
    # TODO: benchmark to figure out better processing_time values for simulation
    timestamp = GlobalTimestamp.remote()
    movie_ff = app.source(
        #DataSource(f"{data_dir}/ratings.csv", timestamp, FLAGS.sleep),
        DataSource(f"{data_dir}/test.csv", FLAGS.sleep),
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
        #User(movie_features, user_features, FLAGS.learning_rate, FLAGS.user_feature_reg),
        operators[FLAGS.update],
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

    print(results_file)
    print("Completed run!")
    log_results(name)


if __name__ == "__main__":
    app.run(main)
##
