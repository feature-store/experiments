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
import wandb

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

class STLLIFO(BaseScheduler):
    def __init__(self, keys: List[str]) -> None:
        self.waker: Optional[threading.Event] = None
        self.stop_iteration = None

        #self.queue = {key: [] for key in keys}
        self.keys = keys
        self.queue: List[Record] = []

    def push_event(self, record: Record):
        self.wake_waiter_if_needed()
        if record.is_stop_iteration():
            #self.stop_iteration = record
            #self.queue.insert(0, record)
            self.queue.append(record)
        else:
            index = -1
            for i in range(len(self.queue)): 
                if not self.queue[i].is_stop_iteration() and self.queue[i].entry.key == record.entry.key: # override
                    self.queue[i] = record
                    index = i 
            if index < 0:
                self.queue.append(record)

            assert len(self.queue) <= len(self.keys), f"Queue is too long {len(self.queue)}, limit: {len(self.keys)}"

    #def choose_key(self): 
    #    start_key = self.key_index
    #    queue = []
    #    while len(queue) == 0:

    #        if self.key_index >= len(list(self.queue.keys())): 
    #            self.key_index = 0 

    #        queue = list(self.queue.keys())[self.key_index]
    #        self.key_index += 1


    def pop_event(self) -> Record:
        if self.stop_iteration: # return stop iteration record
            return self.stop_iteration
        if len(self.queue) == 0:
            return Record.make_wait_event(self.new_waker())

        return self.queue.pop(0)



@dataclass
class SourceValue:
    key: str
    value: int
    timestamp: int
    ingest_time: float


@dataclass
class WindowValue:
    key: str
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
    def __init__(self, data_dir: str, log_filename: str) -> None:

        events_df = pd.read_csv(f"{data_dir}/events.csv")

        self.ts = 0
        self.data = events_df
        self.last_send_time = -1
        self.total = len(events_df.index)

        self.ts_events = dict(tuple(events_df.groupby("timestamp_ms")))
        self.ts_events = {key: self.ts_events[key].to_dict("records") for key in self.ts_events.keys()}
        self.max_ts = events_df.timestamp_ms.max()
        self.total_sent = 0

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
        df = pd.DataFrame({"timestamp_ms": [self.ts], "timestamp": [ingest_time]})
        self._file.write(df.to_csv(index=None, header=None))

        if self.ts % 1000 == 0: 
            print(f"Sent events {ingest_time}: {self.total_sent} / {self.total}")
        self.ts += 1
        time.sleep(0.001)
        return [
            Record(
                entry=SourceValue(
                    key=e["key_id"],
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
        self._data[record.entry.key].append(record.entry.value)

        if len(self._data[record.entry.key]) >= self.window_size:
            window = list(self._data[record.entry.key])
            self._data[record.entry.key] = self._data[record.entry.key][
                self.slide_size :
            ]
            assert (
                len(self._data[record.entry.key]) == self.window_size - self.slide_size
            ), f"List length is wrong size {len(self._data[record.entry.key])}"

            # return window record
            # print("window", record.entry.key, window)
            return Record(
                entry=WindowValue(
                    key=record.entry.key,
                    value=window,
                    timestamp=record.entry.timestamp,
                    ingest_time=record.entry.ingest_time,
                ), 
                shard_key=str(record.entry.key)
            )

class STLFitForecast(BaseTransform):

    """
    Fit STLForecast model and forecast future points. 
    """

    def __init__(self, seasonality, forecast):
        self.seasonality = seasonality
        self.forecast_len = forecast

    def on_event(self, record: Record):
        st = time.time()
        #print("window", len(record.entry.value), record.entry.value[0])
        model = STLForecast(
            np.array(record.entry.value),
            ARIMA, 
            model_kwargs=dict(order=(1, 1, 0), trend="t"), 
            period=self.seasonality
        ).fit() 
        forecast = model.forecast(self.forecast_len)
        timestamp = record.entry.timestamp 
        runtime = st - time.time()
        #print(f"Foreast {record.entry.key}, {runtime}")
        return Record(
            entry=TimeSeriesValue(
                key_id=record.entry.key, 
                forecast=forecast,
                timestamp=record.entry.timestamp,
                ingest_time=record.entry.ingest_time,
                runtime=runtime,
                processing_time=time.time(),
            ), 
            shard_key=str(record.entry.key)
        )

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
    }

    # create feature frames
    # TODO: benchmark to figure out better processing_time values for simulation
    window_ff = app.source(
        DataSource(data_dir, timestamp_file),
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
        STLFitForecast(seasonality=24, forecast=2000),
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
        WriteFeatures(results_file, ["key_id", "forecast", "seasonality", "timestamp", "processing_time", "runtime", "ingest_time"])
    )

    app.deploy()

    if deploy_mode == "simpy":
        env.run(100)
    app.wait()

    query_results_file = f"{results_dir}/{name}_features.csv"
    query_file = f"{data_dir}/queries.csv"

    log_results(name)


if __name__ == "__main__":
    app.run(main)
