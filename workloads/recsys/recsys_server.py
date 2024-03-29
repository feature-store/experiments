from ast import Global
import random
from collections import defaultdict
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


class KeyFIFO(BaseScheduler):
    def __init__(self) -> None:
        self.waker: Optional[threading.Event] = None

        self.start_time = time.time()

        # metric tracking
        self.num_pending = defaultdict(lambda: 0)
        self.last_updated = defaultdict(lambda: self.start_time)
        self.num_updates = defaultdict(lambda: 0)


        # queue
        self.queue = defaultdict(list)
        self.stop_iteration = None

    def push_event(self, record: Record):
        self.wake_waiter_if_needed()

        if record.is_stop_iteration(): 
            self.stop_iteration = record
            return 

        # metric tracking
        key = record.entry.user_id
        self.num_pending[key] += 1

        self.queue[key].append(record)
       
    def choose_key(self): 

        st = time.time()
        last = None
        last_key = None
        for key in self.num_pending.keys(): 

            # no waiting events
            if self.num_pending[key] == 0: continue 

            if last is None or self.last_updated[key] is None or self.last_updated[key] < last: 
                last = self.last_updated[key]
                last_key = key
        print("choose time", time.time() - st)
        return last_key

    def pop_event(self) -> Record:
        if self.stop_iteration:
            return self.stop_iteration
       
        key = self.choose_key()
        if key is None: 
            return Record.make_wait_event(self.new_waker())
        events = self.queue[key]
        self.queue[key] = []

        # metrics 
        self.num_pending[key] = 0
        self.num_updates[key] += len(events)
        self.last_updated[key] = time.time()
        #print("pending", self.num_pending)
        #print("num_updates", self.num_pending)

        return events

class Random(KeyFIFO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_key(self): 
    
        st = time.time()
        keys = []
        for key in self.num_pending.keys():
            if self.num_pending[key] == 0: continue
            keys.append(key)
        if len(keys) == 0: return None

        key = random.choice(keys)
        print("choose time", time.time() - st)
        return key

class Bad(KeyFIFO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_key(self): 
    
        st = time.time()
        keys = []
        for key in self.num_pending.keys():
            if self.num_pending[key] == 0: continue
            keys.append(key)
        if len(keys) == 0: return None

        print("keys", keys)
        key = keys[0]
        print("choose time", time.time() - st)
        return key




class MLFIFO(KeyFIFO):

    def __init__(self, *args, **kwargs):
        dataset_dir = "/data/wooders/ralf-vldb//datasets/ml-latest-small"
        self.model = pickle.load(open(f"{dataset_dir}/random_forest_model", "rb"))
        super().__init__(*args, **kwargs)

    def choose_key(self): 
        t = time.time()
        keys = []
        s = []
        p = []
        u = []
        for key in self.num_pending.keys():
            if self.num_pending[key] == 0: continue

            s.append(t - self.last_updated[key])
            p.append(self.num_pending[key])
            u.append(self.num_updates[key])
            keys.append(key)

        if len(keys) == 0: 
            return None


        max_score = None
        max_key = None

        print(np.array(list(zip(s, u, p))).shape)
        scores = self.model.predict(np.array(list(zip(s, u, p))))
        i = np.argmax(scores)
        print("choose time", time.time() - t)
        return keys[i]

        #for i in range(len(keys)):
        #    # "ML" function
        #    # TODO: replace with actual learned model
        #    #score = self.model.predict([[s[i], u[i], p[i]]])[0]
        #    #print("score", [s[i], p[i], u[i]], "->", score)
        #    #score = s[keys.index(key)] + 10 * p[keys.index(key)]
        #    score = scores[i]

        #    if max_score is None or score > max_score:
        #        max_score = score
        #        max_key = keys[i]
        #return max_key


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
    def __init__(self, file_path: str, log_filename: str, sleep: int = 0, max_ts = None) -> None:
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
        if max_ts is not None: self.max_ts = min(self.max_ts, max_ts)
        self.ts = events_df["timestamp"].min()
        self.data = data
        self.sleep = sleep
        self.last_send_time = -1

        # log when events are sent 
        df = pd.DataFrame({"timestamp_ms": [], "timestamp": []})
        self.filename = log_filename
        df.to_csv(self.filename, index=None)
        self.file = None

    @property
    def _file(self):
        if self.file is None:
            self.file = open(self.filename, "a")
        return self.file





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
        t = time.time()

        # log timestamps 
        df = pd.DataFrame({"timestamp": [curr_timestamp], "time": [t]})
        self._file.write(df.to_csv(index=None, header=None))


        # aggregate multiple updates into single list
        movie_ids = defaultdict(list)
        ratings = defaultdict(list)
        for e in events:
            movie_ids[e["user_id"]].append(e["movie_id"])
            ratings[e["user_id"]].append(e["rating"])

        return [
            Record(
                entry=SourceValue(
                    movie_id=movie_ids[user_id],
                    user_id=user_id, 
                    timestamp=curr_timestamp,
                    rating=ratings[user_id],
                    ingest_time=t, 
                ),
                shard_key=str(user_id)
            ) for user_id in movie_ids.keys()
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

        # Note: updates take about 0.3-0.35s


    def on_events(self, records: List[Record]):

        movie_ids = []
        ratings = []

        user = records[0].entry.user_id
        timestamp = records[0].entry.timestamp
        ingest_time = records[0].entry.ingest_time

        for record in records:
            assert record.entry.user_id == user
            movie_ids  += record.entry.movie_id
            ratings += record.entry.rating

            if record.entry.timestamp > timestamp: timestamp = record.entry.timestamp
            if record.entry.ingest_time > record.entry.ingest_time: ingest_time = record.entry.ingest_time

        return self.on_event(Record(
            entry=SourceValue(
                user_id=user, 
                movie_id=movie_ids, 
                rating=ratings, 
                timestamp=timestamp, 
                ingest_time=ingest_time
            )
        ))


    def on_event(self, record: Record):
        ui = self.user_to_index[str(record.entry.user_id)]
        st = time.time()

        # required to make things writable... idk why
        self.A = np.array(self.A)
        self.R = np.array(self.R)
        self.user_matrix = np.array(self.user_matrix)
        self.movie_matrix = np.array(self.movie_matrix)

        # update matrix
        for i in range(len(record.entry.movie_id)):
            mi = self.movie_to_index[str(record.entry.movie_id[i])]
            rating = record.entry.rating[i]
            self.A[ui][mi] = rating
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
    name = f"results_{FLAGS.update}_workers_{FLAGS.workers}_{FLAGS.scheduler}_learningrate_{FLAGS.learning_rate}_userfeaturereg_{FLAGS.user_feature_reg}_sleep_{FLAGS.sleep}"
    print("dataset", data_dir)

    ## create results file/directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/{name}.csv"
    timestamp_file = f"{results_dir}/timestamps/{name}.csv"
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
        "key-fifo": KeyFIFO(), 
        "ml": MLFIFO(), 
        "least": LeastUpdate(),
        "random": Random(),
        "bad": Bad(), 
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
        DataSource(f"{data_dir}/test.csv", timestamp_file, FLAGS.sleep, max_ts=40000),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
                stop_after_s=10
            ),         
            ray_config=RayOperatorConfig(num_replicas=1)
    )
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
