from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
from typing import List
import os
import time
from collections import defaultdict
import pandas as pd
import simpy
from statsmodels.tsa.seasonal import STL
from absl import app, flags
import wandb

# might need to do  export PYTHONPATH='.'
from workloads.util import WriteFeatures

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
    trend: float
    seasonality: List[float]
    timestamp_ms: int
    ingest_time: float
    processing_time: int
    runtime: float


class DataSource(BaseTransform):
    def __init__(self, data_dir: str) -> None:

        events_df = pd.read_csv(f"{data_dir}/events.csv")

        self.ts = 0
        self.data = events_df
        self.last_send_time = -1

    def on_event(self, _: Record) -> List[Record[SourceValue]]:

        # TODO: fix this - very slow
        events = self.data[self.data["timestamp_ms"] == self.ts].to_dict("records")

        num_remaining = len(self.data[self.data["timestamp_ms"] >= self.ts].index)
        if num_remaining == 0:
            raise StopIteration()
        ingest_time = time.time()
        if len(events) > 0:
            print("sending events", self.ts, len(events), "remaining", num_remaining)
        self.ts += 1
        return [
            Record(
                SourceValue(
                    key=e["key_id"],
                    value=e["value"],
                    timestamp=e["timestamp_ms"],
                    ingest_time=ingest_time,
                )
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
                WindowValue(
                    key=record.entry.key,
                    value=window,
                    timestamp=record.entry.timestamp,
                    ingest_time=record.entry.ingest_time,
                )
            )


class STLFit(BaseTransform):
    def __init__(self):
        self.seasonality = 12

    def on_event(self, record: Record):
        st = time.time()
        stl_result = STL(record.entry.value, period=self.seasonality, robust=True).fit()
        # print(time.time() - st, st)
        trend = stl_result.trend[-1]
        seasonality = list(stl_result.seasonal[-(self.seasonality + 1) : -1])
        # print(record.entry.key, trend, seasonality)

        return Record(
            TimeSeriesValue(
                key_id=record.entry.key,
                trend=trend,
                seasonality=seasonality,
                timestamp_ms=record.entry.timestamp,
                ingest_time=record.entry.ingest_time,
                processing_time=time.time(),
                runtime=time.time() - st,
            )
        )


#class WriteFeatures(BaseTransform): 
#    def __init__(self, filename: str):
#        df = pd.DataFrame({"key_id": [], "trend": [], "seasonality": [], "timestamp_ms": [], "processing_time": [], "runtime": [], "ingest_time": []})
#        self.filename = filename 
#        df.to_csv(self.filename, index=None)
#        #self.file = open(self.filename, "a")
#
#    def on_event(self, record: Record): 
#        #row = ','.join([str(col) for col in [record.entry.key, record.entry.trend, record.entry.seasonality, record.entry.timestamp, record.entry.processing_time, record.entry.runtime]]) + "\n"
#        df = pd.DataFrame([record.entry.key, record.entry.trend, record.entry.seasonality, record.entry.timestamp, record.entry.processing_time, record.entry.runtime, record.entry.ingest_time])
#        open(self.filename, "a").write(df.T.to_csv(index=None, header=None))
#        #print("wrote", df.T.to_csv())
#        print("wrote", record.entry.key, record.entry.timestamp)
        
#class WriteFeatures(BaseTransform):
#    def __init__(self, filename: str):
#        df = pd.DataFrame(
#            {
#                "key_id": [],
#                "trend": [],
#                "seasonality": [],
#                "timestamp_ms": [],
#                "processing_time": [],
#                "runtime": [],
#                "ingest_time": [],
#            }
#        )
#        self.filename = filename
#        df.to_csv(self.filename, index=None)
#        self.file = None
#
#    @property
#    def _file(self):
#        if self.file is None:
#            self.file = open(self.filename, "a")
#        return self.file
#
#    def on_event(self, record: Record):
#        # row = ','.join([str(col) for col in [record.entry.key, record.entry.trend, record.entry.seasonality, record.entry.timestamp, record.entry.processing_time, record.entry.runtime]]) + "\n"
#        df = pd.DataFrame(
#            [
#                record.entry.key,
#                record.entry.trend,
#                record.entry.seasonality,
#                record.entry.timestamp,
#                record.entry.processing_time,
#                record.entry.runtime,
#                record.entry.ingest_time,
#            ]
#        )
#        self._file.write(df.T.to_csv(index=None, header=None))
#        # print("wrote", df.T.to_csv())
#        print("wrote", record.entry.key, record.entry.timestamp)


def main(argv):
    print("Running STL pipeline on ralf...")

    if FLAGS.source_dir is None:
        # download data
        run = wandb.init(project="ralf-stl", entity="ucb-ralf")
        src_artifact = run.use_artifact(f"{FLAGS.experiment}:latest", type="dataset")
        data_dir = src_artifact.download()
        print(data_dir)
    else:
        # use existing data dir
        data_dir = FLAGS.source_dir

    # create results file/directory
    results_dir = f"{FLAGS.target_dir}/{FLAGS.experiment}"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/results_workers_{FLAGS.workers}_{FLAGS.scheduler}_window_{FLAGS.window_size}_slide_{FLAGS.slide_size}.csv"

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
        "lifo": LIFO(),
    }

    # create feature frames
    # TODO: benchmark to figure out better processing_time values for simulation
    window_ff = app.source(
        DataSource(data_dir),
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
        STLFit(),
        scheduler=schedulers[FLAGS.scheduler],
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.2, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers)
        )
    ).transform(
        WriteFeatures(results_file, ["key_id", "trend", "seasonality", "timestamp_ms", "processing_time", "runtime", "ingest_time"])
    )

    app.deploy()

    if deploy_mode == "simpy":
        env.run(100)
    app.wait()

    print("logging to wandb")
    if FLAGS.source_dir:
        run = wandb.init(project="ralf-stl", entity="ucb-ralf")
    target_artifact = wandb.Artifact(f"{FLAGS.experiment}-results", type="results")
    target_artifact.add_dir(results_dir)
    run.log_artifact(target_artifact)
    print("Completed run!")


if __name__ == "__main__":
    app.run(main)
