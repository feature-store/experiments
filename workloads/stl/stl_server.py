from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
from typing import List, Optional
import os
import time
from collections import defaultdict
import pandas as pd
import simpy
from statsmodels.tsa.seasonal import STL
from absl import app, flags
import wandb

from ralf.v2.utils import get_logger

# might need to do  export PYTHONPATH='.'
from workloads.util import (
    read_config,
    use_dataset,
    log_dataset,
    log_results,
    WriteFeatures,
)

logger = get_logger()

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
    trend: List[float]
    seasonality: List[float]
    timestamp_ms: int
    ingest_time: float
    processing_time: int
    runtime: float


class DataSource(BaseTransform):
    def __init__(self, data_dir: str) -> None:
        path = f"{data_dir}/events.csv"
        events_df = pd.read_csv(path)
        logger.msg(f"read data from path {path}")

        # TODO(simon): remove this, used by debugging only
        # events_df = events_df[events_df["key_id"] == 1]

        self.ts = -5
        self.data = events_df
        self.last_send_time = -1
        self.total = len(events_df.index)

        self.ts_events = dict(tuple(events_df.groupby("timestamp_ms")))
        # print(self.ts_events.keys())
        self.max_ts = events_df.timestamp_ms.max()

    def on_event(self, _: Record) -> Optional[List[Record[SourceValue]]]:
        self.ts += 5

        if self.ts > self.max_ts:
            raise StopIteration()

        if self.ts not in self.ts_events:
            # TODO(simon): source will immediately poll next, should we sleep here?
            # TODO(simon): onsider back pressure
            logger.msg(f"Skipping ts {self.ts} because not in self.ts_events")
            return None

        events = self.ts_events[self.ts]

        # TODO(simon): make sure we no longer need this with event_metrics
        ingest_time = time.time()
        return [
            Record(
                SourceValue(
                    key=e["key_id"],
                    value=e["value"],
                    timestamp=e["timestamp_ms"],
                    ingest_time=ingest_time,
                ),
                shard_key=str(e["key_id"]),
            )
            for _, e in events.iterrows()
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
                ),
                shard_key=str(record.entry.key),
            )


class STLFit(BaseTransform):
    def __init__(self):
        self.seasonality = 12

    def on_event(self, record: Record):
        st = time.time()
        stl_result = STL(record.entry.value, period=self.seasonality, robust=True).fit()
        # print(time.time() - st, st)
        trend = list(stl_result.trend)
        # TODO: potentially interpolate trend?
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


def main(argv):
    print("Running STL pipeline on ralf...")

    data_dir = use_dataset(FLAGS.experiment, redownload=False)
    results_dir = os.path.join(read_config()["results_dir"], FLAGS.experiment)
    name = f"results_workers_{FLAGS.workers}_{FLAGS.scheduler}_window_{FLAGS.window_size}_slide_{FLAGS.slide_size}"
    print("Using data from", data_dir)
    print("Making results for", results_dir)

    ## create results file/directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/{name}.csv"
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
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers),
        ),
    ).transform(
        WriteFeatures(
            results_file,
            [
                "key_id",
                "trend",
                "seasonality",
                "timestamp_ms",
                "processing_time",
                "runtime",
                "ingest_time",
            ],
        )
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
