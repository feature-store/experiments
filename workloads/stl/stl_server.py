from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record, BaseScheduler
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
import numpy as np
from typing import List
import os
import time
from collections import defaultdict
import pandas as pd
import simpy
from statsmodels.tsa.seasonal import STL
from absl import app, flags
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sktime.performance_metrics.forecasting import mean_squared_scaled_error
import wandb
import warnings

# might need to do  export PYTHONPATH='.'
from workloads.util import read_config, use_dataset, log_dataset, log_results, WriteFeatures

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
    "window_size",
    default=None,
    help="Window size",
    required=True,
)


flags.DEFINE_integer(
    "slide_size",
    default=None,
    help="Slide size",
    required=True,
)

flags.DEFINE_integer(
    "workers",
    default=2,
    help="Number of workers for bottlenck operator",
    required=False,
)

flags.DEFINE_string(
    "source_dir",
    default=None,
    help="Dataset directory",
    required=False,
)

flags.DEFINE_string(
    "target_dir",
    default="./results",
    help="Result target directory",
    required=False,
)

class CumulativeErrorScheduler(BaseScheduler):
    """Order events by prioritizing keys with the largest cumulative error
    """

    def __init__(self, keys: List[int]) -> None:
        self.waker: Optional[threading.Event] = None
        self.stop_iteration = None

        self.queue = defaultdict(lambda: None)
        self.keys = keys

        # priority tracking 
        self.priority = [0] * (max([int(key) for key in keys]) + 1)
        self.predictions = defaultdict(list)
        self.values = defaultdict(list)
        self.max_prio = 10000000

    def update_priority(self, record):
        key = record.entry.key_id
        ts = record.entry.timestamp
        value = record.entry.value[-1]

        # run prediction
        feature = self._operator.get(key)
        if feature is None: 
            # TODO: make sure this doesn't happen
            assert False, f"Feature {key} not set"
            self.priority[key] = self.max_prio
            return 

        forecast = feature.entry.forecast
        model_ts = feature.entry.timestamp
        ts_delta = ts - model_ts
        if ts_delta >= len(forecast):
            # TODO: make sure this doesn't happen
            self.priority[key] = self.max_prio
            return 

        assert ts_delta >= 0, f"Model is has newer timestamp {model_ts}, {ts} - key {key}"
        self.predictions[key].append(forecast[-(ts_delta)])
        self.values[key].append(value)

        assert len(self.values[key]) == len(self.predictions[key]), f"Mismatched lengths"

        if len(self.values[key]) <= 1: 
            # not enough points
            return 

        # calculate error
        error = mean_absolute_scaled_error(
            np.array(self.values[key]), 
            np.array(self.predictions[key]), 
            y_train=np.array(self.values[key])
        )
        self.priority[key] += error 

    def choose_key(self): 
        """Choose key to re-compute feature for.

        :return: key to update next
        :rtype: str
        """

        key = np.array(self.priority).argmax()

        # clear errors 
        self.priority[key] = 0
        self.predictions[key] = []
        self.values[key] = []
       
        #return key 
        return key

    def push_event(self, record: Record):
        self.wake_waiter_if_needed()
        if record.is_stop_iteration():
            self.stop_iteration = record
        else:
            # override with new window
            self.queue[int(record.entry.key_id)] = record
            #print("Window", record.entry.key_id, record.entry.timestamp)

            # update priority
            self.update_priority(record)
            print(self.priority)

    def pop_event(self) -> Record:
        if self.stop_iteration: # return stop iteration record
            print("Return STOP")
            return self.stop_iteration

        key = self.choose_key()
        #print(self.queue)
        if self.queue[key] is None: 
            return Record.make_wait_event(self.new_waker())

        print("Pending", len([v for v in self.queue.values() if v is not None]))
        event = self.queue[key]
        self.queue[key] = None
        return event



@dataclass
class SourceValue:
    key_id: str
    value: int
    timestamp: int
    ingest_time: float


@dataclass
class WindowValue:
    key_id: str
    value: List[int]
    timestamp: int
    ingest_time: float


@dataclass
class TimeSeriesValue:
    key_id: str
    forecast: List[float]
    timestamp: int
    ingest_time: float
    runtime: float
    processing_time: float
 

class DataSource(BaseTransform):
    def __init__(self, data_dir: str, log_filename: str, sleep: float, window_size: int) -> None:

        events_df = pd.read_csv(f"{data_dir}/events.csv")

        self.ts = window_size # start at last window
        self.data = events_df
        self.last_send_time = -1
        self.total = len(events_df.index)

        self.ts_events = dict(tuple(events_df.groupby("timestamp_ms")))
        self.ts_events = {key: self.ts_events[key].to_dict("records") for key in self.ts_events.keys()}
        self.max_ts = events_df.timestamp_ms.max()
        self.total_sent = 0
        self.sleep = sleep

        # log when events are sent 
        df = pd.DataFrame({"timestamp": [], "processing_time": []})
        self.filename = log_filename
        df.to_csv(self.filename, index=None)

    def on_event(self, _: Record) -> List[Record[SourceValue]]:

        if self.ts >= self.max_ts: 
            print("completed iteration", self.total_sent)
            raise StopIteration()
        else: 
            events = []
            if self.ts in self.ts_events: 
                events = self.ts_events[self.ts]
                #print(events)
            self.total_sent += len(events)


        ingest_time = time.time()

        # log timestamps 
        df = pd.DataFrame({"timestamp": [self.ts], "processing_time": [ingest_time]})
        df.to_csv(self.filename, mode="a", index=False, header=False)

        if self.ts % 1000 == 0: 
            print(f"Sent events {ingest_time}: {self.total_sent} / {self.total}")
        self.ts += 1
        time.sleep(self.sleep)
        print("TIMESTAMP", self.ts)
        return [
            Record(
                entry=SourceValue(
                    key_id=e["key_id"],
                    value=e["value"],
                    timestamp=e["timestamp_ms"],
                    ingest_time=ingest_time,
                ), 
                shard_key=str(e["key_id"])
            )
            for e in events
        ]


class Window(BaseTransform):
    def __init__(self, window_size, slide_size=None) -> None:
        self._data = defaultdict(list)
        self.window_size = window_size
        self.slide_size = window_size if slide_size is None else slide_size


    def on_event(self, record: Record):
        self._data[record.entry.key_id].append(record.entry.value)

        if len(self._data[record.entry.key_id]) >= self.window_size:
            window = list(self._data[record.entry.key_id])
            self._data[record.entry.key_id] = self._data[record.entry.key_id][
                self.slide_size :
            ]
            assert (
                len(self._data[record.entry.key_id]) == self.window_size - self.slide_size
            ), f"List length is wrong size {len(self._data[record.entry.key_id])}"

            # return window record
            # print("window", record.entry.key, window)
            return Record(
                entry=WindowValue(
                    key_id=record.entry.key_id,
                    value=window,
                    timestamp=record.entry.timestamp,
                    ingest_time=record.entry.ingest_time,
                ), 
                shard_key=str(record.entry.key_id)
            )

class STLFitForecast(BaseTransform):

    """
    Fit STLForecast model and forecast future points. 
    """

    def __init__(self, seasonality, forecast, window_size: int, data_dir: str):
        self.seasonality = seasonality
        self.forecast_len = forecast
        #self.data = defaultdict(lambda: None)
        self.data = self.init_data(data_dir, window_size)

    def init_data(self, data_dir, window_size):
        events_df = pd.read_csv(f"{data_dir}/events.csv")
        events = dict(tuple(events_df.groupby("key_id")))
        data = {}
        for key_id in events.keys(): 
            print(events[key_id])
            window = [e["value"] for e in events[key_id][:window_size]]
            timestamp = events[key_id][window_size-1]["timestamp_ms"]
            ingest_time = None
            record = self.fit_model(key_id, window, timestamp, ingest_time)
            data[key_id] = record
        print("init data", data)
        return data


    def get(self, key): 
        return self.data[key]

    def fit_model(self, key_id, window, timestamp, ingest_time):
        st = time.time()
        # catch warning for ML fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = STLForecast(
                np.array(window),
                ARIMA, 
                model_kwargs=dict(order=(1, 1, 0), trend="t"), 
                period=self.seasonality
            ).fit() 
            forecast = model.forecast(self.forecast_len)
        runtime = time.time() - st
        return Record(
            entry=TimeSeriesValue(
                key_id=key_id, 
                forecast=forecast.tolist(),
                timestamp=timestamp,
                runtime=runtime,
                processing_time=time.time(),
            ), 
            shard_key=str(key_id)
        )

    def on_event(self, record: Record):
        result_record = self.fit_model(
            record.entry.key_id, 
            record.entry.value, 
            record.entry.timestamp, 
            record.entry.ingest_time
        )
        self.data[record.entry.key_id] = result_record
        return result_record

#class STLFit(BaseTransform):
    #def __init__(self):
        #self.seasonality = 12

    #def on_event(self, record: Record):
        #st = time.time()
        #stl_result = STL(record.entry.value, period=self.seasonality, robust=True).fit()
        ## print(time.time() - st, st)
        #trend = list(stl_result.trend)
        ## TODO: potentially interpolate trend? 
        #seasonality = list(stl_result.seasonal[-(self.seasonality + 1) : -1])
        ## print(record.entry.key, trend, seasonality)

        #return Record(
            #entry=TimeSeriesValue(
                #key_id=record.entry.key,
                #trend=trend,
                #seasonality=seasonality,
                #timestamp_ms=record.entry.timestamp,
                #ingest_time=record.entry.ingest_time,
                #processing_time=time.time(),
                #runtime=time.time() - st,
            #),
            #shard_key=str(record.entry.key)
        #)

def main(argv):
    print("Running STL pipeline on ralf...")

    data_dir = use_dataset(FLAGS.experiment, download=False)
    results_dir = os.path.join(read_config()["results_dir"], FLAGS.experiment)
    name = f"results_workers_{FLAGS.workers}_{FLAGS.scheduler}_window_{FLAGS.window_size}_slide_{FLAGS.slide_size}"
    print("Using data from", data_dir)
    print("Making results for", results_dir)

    ## create results file/directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/{name}.csv"
    timestamp_file = f"{results_dir}/{name}_timestamps.csv"
    print("results file", results_file)

    # deploy_mode = "ray"
    deploy_mode = "ray"
    # deploy_mode = "simpy"
    app = RalfApplication(RalfConfig(deploy_mode=deploy_mode))

    # create simulation env
    if deploy_mode == "simpy":
        env = simpy.Environment()
    else:
        env = None

    schedulers = {
        "fifo": FIFO(),
        "lifo": LIFO(), #STLLIFO(keys=range(1, 101, 1)),
        "ce": CumulativeErrorScheduler(keys=range(1, 101, 1)),
    }

    # create feature frames
    # TODO: benchmark to figure out better processing_time values for simulation
    window_ff = app.source(
        DataSource(data_dir, timestamp_file, sleep=1, window_size=FLAGS.window_size),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, processing_time_s=0.01, stop_after_s=10
            ),
            ray_config=RayOperatorConfig(num_replicas=1),
        ),
    ).transform(
        Window(window_size=FLAGS.window_size, slide_size=FLAGS.slide_size),
        scheduler=FIFO(),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env,
                processing_time_s=0.01,
            ),
            ray_config=RayOperatorConfig(num_replicas=2),
        ),
    )
    stl_ff = window_ff.transform(
        STLFitForecast(seasonality=24, forecast=2000, window_size=FLAGS.window_size, data_dir=data_dir),
        scheduler=schedulers[FLAGS.scheduler],
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.2, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers)
        )
    ).transform(
        #WriteFeatures(results_file, ["key_id", "trend", "seasonality", "timestamp_ms", "processing_time", "runtime", "ingest_time"])
        WriteFeatures(results_file, ["key_id", "forecast", "timestamp", "processing_time", "runtime", "ingest_time"])
    )

    app.deploy()

    if deploy_mode == "simpy":
        env.run(100)
    app.wait()

    query_results_file = f"{results_dir}/{name}_features.csv"
    query_file = f"{data_dir}/queries.csv"

    print(f"See results {results_file}")
    log_results(name)


if __name__ == "__main__":
    app.run(main)
