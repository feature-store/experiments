import itertools
import json
import os
import pickle
import sqlite3
import threading
import time
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional

import numpy as np
from absl import app, flags
from ralf.v2 import (
    FIFO,
    BaseScheduler,
    BaseTransform,
    RalfApplication,
    RalfConfig,
    Record,
)
from ralf.v2.operator import OperatorConfig, RayOperatorConfig
from ralf.v2.utils import get_logger
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from sortedcontainers import SortedSet

logger = get_logger()

FLAGS = flags.FLAGS

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
    default=None,
    help="Number of workers for bottlenck operator",
    required=True,
)
flags.DEFINE_string(
    "azure_database",
    default=None,
    help="Azure time series database",
    required=True,
)

flags.DEFINE_string(
    "results_dir",
    default=f"results/stl/results/{int(time.time())}",
    help="Diretory to write result jsonl to",
    required=False,
)

flags.DEFINE_float(
    "epsilon",
    default=None,
    help="Default error to add to each time a query is made",
    required=False,
)

KeyType = int


class BasePriorityScheduler(BaseScheduler):
    def __init__(self):
        self.key_to_event: Dict[KeyType, Record] = dict()
        # set initial priority scores to infinity
        self.key_to_priority: Dict[KeyType, float] = dict()
        self.sorted_keys_by_prio = SortedSet(key=lambda key: self.key_to_priority[key])
        self.max_prio = 100000000000
        self.keys = set([])

        self.waker: Optional[threading.Event] = None
        self.stop_iteration = None
        self._writer_lock = None

    @property
    def writer_lock(self):
        if self._writer_lock is None:
            self._writer_lock = threading.Lock()
        return self._writer_lock

    @abstractmethod
    def compute_priority(self, record: Record) -> float:
        raise NotImplementedError()

    def _compute_priority_wrapper(self, record: Record) -> float:
        feature: Optional[Record] = self._operator.get(record.shard_key)
        if feature is None:
            logger.msg(
                f"Missing feature for key {record.shard_key}, returning max_prio"
            )
            return self.max_prio
        return self.compute_priority(record)

    def push_event(self, record: Record):
        if record.is_stop_iteration():
            self.stop_iteration = record
            self.wake_waiter_if_needed()
            return

        record_key = record.shard_key
        self.keys.add(record_key)
        with self.writer_lock:
            self.key_to_event[record_key] = record
            if record_key in self.sorted_keys_by_prio:
                self.sorted_keys_by_prio.remove(record_key)
            self.key_to_priority[record_key] = self._compute_priority_wrapper(record)
            self.sorted_keys_by_prio.add(record_key)
        self.wake_waiter_if_needed()

    def pop_event(self) -> Record:
        if self.stop_iteration:
            return self.stop_iteration

        with self.writer_lock:
            if len(self.key_to_event) == 0:
                return Record.make_wait_event(self.new_waker())

            latest_key = self.sorted_keys_by_prio.pop()
            record = self.key_to_event.pop(latest_key)
            self.key_to_priority.pop(latest_key)
        return record

    def qsize(self) -> int:
        return len(self.key_to_event)


class KeyAwareLifo(BasePriorityScheduler):
    """Always prioritize the latest record by arrival time, if the record has"""

    def compute_priority(self, _: Record) -> float:
        return time.time()


class RoundRobinScheduler(BasePriorityScheduler):
    """Prioritize the key that hasn't been updated for longest."""

    def compute_priority(self, record: Record) -> float:
        return self.key_to_priority.get(record.shard_key, 0) + 1


class CumulativeErrorScheduler(BasePriorityScheduler):
    """Prioritize the key that has highest prediction error so far"""

    def __init__(self, epsilon=None):
        # TODO: bring back the logic that temporarily disable a key if it is pending update
        # If that ever becomes an issue.
        # self.pending_updates: Dict[KeyType, float] = []

        super().__init__()
        self.epsilon = epsilon
        self.last_seqno = {}

    def compute_priority(self, record: Record["WindowValue"]) -> float:
        assert isinstance(record.entry, WindowValue)

        # start from last seen seqno
        start = self.last_seqno.get(record.shard_key, -1)
        incoming_seqnos = np.array([n for n in record.entry.seq_nos if n > start])
        self.last_seqno[record.shard_key] = incoming_seqnos.max()
        # print("length", incoming_seqnos.shape, start)

        # lookup current feature
        feature: Record[TimeSeriesValue] = self._operator.get(record.shard_key)

        forecast = np.array(feature.forecast)
        window_last_seqno = feature.last_seqno
        forecast_indicies = incoming_seqnos - window_last_seqno - 1

        positive_indicies = np.argwhere(forecast_indicies >= 0)
        forecast_indicies = np.take(forecast_indicies, positive_indicies)

        y_true = np.take(record.entry.values, positive_indicies)
        y_pred = np.take(forecast, forecast_indicies)
        y_train = np.array(feature.y_train)

        assert (
            len(record.entry.seq_nos) == 864
        ), f"Unexpected length {len(record.entry.seq_nos)}"

        # TODO: sample if too heavy weight
        # TODO: maybe scale this by staleness
        error = mean_absolute_scaled_error(
            y_true=y_true, y_pred=y_pred, y_train=y_train
        ) * len(y_true)

        if self.epsilon is not None:
            # return max of adding epsilon or returning ASE
            return max(
                self.key_to_priority.get(record.shard_key, 0)
                + self.epsilon,  # add epsilon each time
                error,
            )
        return error


@dataclass
class SourceValue:
    key_id: int
    value: float
    seq_no: int
    ingest_time: float


class DataSource(BaseTransform):
    """Generate event data over keys"""

    def __init__(
        self,
        keys: List[int],
        sleep: float,
        results_dir: str,
        azure_database: str,
        all_timestamps: List[int],
    ):

        self.keys = keys
        self.sleep = sleep
        self.ts = -1
        self.results_dir = results_dir
        self.azure_database = azure_database
        self.all_timestamps = all_timestamps

        self.keys_selection_clause = ",".join(map(str, self.keys))

    def prepare(self):
        self.conn = sqlite3.connect(self.azure_database)
        self.conn.executescript("PRAGMA journal_mode=WAL;")
        logger.msg(f"Total number of timestamps: {len(self.all_timestamps)}")
        logger.msg(f"Total number of keys: {len(self.keys)}")

        cache_path = f"data_cache_keys_{len(self.keys)}.pkl"
        if os.path.exists(cache_path):
            logger.msg(f"Loading from cache {cache_path}")
            with open(cache_path, "rb") as f:
                self.buffer = pickle.load(f)
        else:
            logger.msg(f"Generating fresh cache {cache_path}")
            cursor = self.conn.execute(
                "SELECT timestamp, int_id, avg_cpu FROM readings WHERE "
                f"int_id in ({self.keys_selection_clause})"
            )
            self.buffer = defaultdict(list)
            for ts, int_id, avg_cpu in cursor:
                self.buffer[ts].append((int_id, avg_cpu))
            with open(cache_path, "wb") as f:
                pickle.dump(self.buffer, f)
        logger.msg("All keys loaded")
        logger.msg(f"Num timestamps in self.buffer {len(self.buffer)}")

        self.result_file = open(
            os.path.join(self.results_dir, f"source.{os.getpid()}.jsonl"), "w"
        )

    def on_event(self, _: Record) -> List[Record[SourceValue]]:
        if self.sleep > 0:
            time.sleep(self.sleep)
        self.ts += 1

        if self.ts >= len(self.all_timestamps):
            raise StopIteration()
        azure_ts = self.all_timestamps[self.ts]
        if azure_ts not in self.buffer:
            return
        batch_data = self.buffer.pop(azure_ts)

        ingest_time = time.time()

        batch = [
            Record(
                entry=SourceValue(
                    key_id=key_id,
                    value=avg_cpu,
                    seq_no=self.ts,
                    ingest_time=ingest_time,
                ),
                shard_key=str(key_id),
            )
            for key_id, avg_cpu in batch_data
        ]
        if len(batch) == 0:
            return

        # print(f"Sending {len(batch)} rows at {self.ts} at {ingest_time}")

        self.result_file.write(json.dumps([i.entry.__dict__ for i in batch]))
        self.result_file.write("\n")
        self.result_file.flush()
        return batch


@dataclass
class WindowValue:
    key_id: int
    values: List[float]
    seq_nos: List[int]
    last_ingest_time: float


class Window(BaseTransform):
    def __init__(self, window_size: int, slide_size: int) -> None:
        self._data: DefaultDict[int, List[float]] = defaultdict(list)
        self._seq_nos: DefaultDict[int, List[int]] = defaultdict(list)
        self.window_size = window_size
        self.slide_size = slide_size
        self.st = defaultdict(lambda: 0)

    def prepare(self):
        logger.msg(
            f"Working with window_size {self.window_size} slide_size {self.slide_size}"
        )

    def on_event(self, record: Record[SourceValue]) -> Optional[Record[WindowValue]]:
        key_id = record.entry.key_id
        self._data[key_id].append(record.entry.value)
        self._seq_nos[key_id].append(record.entry.seq_no)

        if len(self._data[key_id]) >= self.window_size:
            # print("window time", time.time() - st[key_id])
            self.st[key_id] = time.time()

            window = list(self._data[key_id])
            self._data[key_id] = self._data[key_id][self.slide_size :]
            seq_nos = list(self._seq_nos[key_id])
            self._seq_nos[key_id] = self._seq_nos[key_id][self.slide_size :]

            assert len(window) == self.window_size, f"Invalid window length {window}"

            return Record(
                entry=WindowValue(
                    key_id=key_id,
                    values=window,
                    seq_nos=seq_nos,
                    last_ingest_time=record.entry.ingest_time,
                ),
                shard_key=str(record.entry.key_id),
            )
        return None


@dataclass
class TimeSeriesValue:
    key_id: int
    forecast: List[float]
    last_seqno: int
    last_ingest_time: float
    processing_time: float
    y_train: List[float]


class STLFitForecast(BaseTransform):
    """
    Fit STLForecast model and forecast future points.
    """

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.start_time = None
        self.num_updates = 0

    def prepare(self):
        self.data = defaultdict(lambda: None)
        self.result_file = open(
            os.path.join(self.results_dir, f"forecast.{os.getpid()}.jsonl"), "w"
        )

    def get(self, key):
        return self.data[key]

    def on_event(self, record: Record[WindowValue]):
        key_id = record.shard_key

        if self.start_time is None:
            self.start_time = time.time()

        with warnings.catch_warnings():
            # catch warning for ML fit
            warnings.filterwarnings("ignore")
            st = time.time()
            model = STLForecast(
                np.array(record.entry.values),
                ARIMA,
                model_kwargs=dict(order=(1, 1, 0), trend="t"),
                period=12 * 24,  # 5 min timestamp interval, period of one day
            ).fit()
            forecast = model.forecast(9000)
            print("runtime", time.time() - st)

        self.num_updates += 1
        if self.num_updates % 1000:
            print(
                "avg throughput",
                self.num_updates,
                self.num_updates / (time.time() - self.start_time),
            )

        self.num_updates += 1
        print(
            "avg throughput",
            self.num_updates,
            self.num_updates / (time.time() - self.start_time),
        )

        forecast_record = TimeSeriesValue(
            key_id=key_id,
            forecast=forecast.tolist(),
            last_seqno=record.entry.seq_nos[-1],
            last_ingest_time=record.entry.last_ingest_time,
            processing_time=time.time(),
            y_train=record.entry.values,
        )

        self.data[key_id] = forecast_record
        # print(f"Update key {key_id}, last_seq_no {record.entry.seq_nos[-1]}")

        self.result_file.write(
            json.dumps(
                {
                    f: forecast_record.__dict__[f]
                    for f in [
                        "key_id",
                        "forecast",
                        "last_seqno",
                        "last_ingest_time",
                        "processing_time",
                    ]
                }
            )
        )
        self.result_file.write("\n")
        self.result_file.flush()


def _get_config() -> Dict:
    """Return all the flag vlaue defined here."""
    return {f.name: f.value for f in FLAGS.get_flags_for_module("__main__")}


flags.DEFINE_integer(
    "num_keys", default=None, required=True, help="limit number of keys"
)

flags.DEFINE_float(
    "source_sleep_per_batch",
    default=None,
    required=True,
    help="source sleep duration in s",
)


def main(argv):
    logger.msg("Running STL pipeline on ralf...")
    print("Results", f"{FLAGS.results_dir}/metrics")

    # Setup dataset directory
    conn = sqlite3.connect(FLAGS.azure_database)
    conn.executescript("PRAGMA journal_mode=WAL;")

    num_keys = FLAGS.num_keys
    cache_file = f"query_cache_{num_keys}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
            keys = cache["keys"]
            all_timestamps = cache["all_timestamps"]
    else:
        logger.msg(f"Genering query cache for num_keys={num_keys}")
        keys = list(
            itertools.chain.from_iterable(
                conn.execute(
                    f"SELECT int_id FROM readings GROUP BY int_id LIMIT {num_keys};"
                )
            )
        )
        all_timestamps = list(
            itertools.chain.from_iterable(
                conn.execute(
                    "SELECT timestamp FROM readings GROUP BY timestamp ORDER BY timestamp"
                ).fetchall()
            )
        )
        with open(cache_file, "w") as f:
            json.dump({"keys": keys, "all_timestamps": all_timestamps}, f)
    logger.msg(f"Working with {len(keys)} keys")

    os.makedirs(FLAGS.results_dir)
    os.makedirs(f"{FLAGS.results_dir}/metrics")
    logger.msg(f"Results dir {FLAGS.results_dir}")
    with open(f"{FLAGS.results_dir}/config.json", "w") as f:
        json.dump(_get_config(), f)

    app = RalfApplication(
        RalfConfig(deploy_mode="ray", metrics_dir=f"{FLAGS.results_dir}/metrics")
    )

    # scheduler options
    schedulers = {
        "lifo": KeyAwareLifo(),
        "rr": RoundRobinScheduler(),
        "ce": CumulativeErrorScheduler(FLAGS.epsilon),
    }

    app.source(
        DataSource(
            keys=keys,
            sleep=FLAGS.source_sleep_per_batch,
            results_dir=FLAGS.results_dir,
            azure_database=FLAGS.azure_database,
            all_timestamps=all_timestamps,
        ),
        operator_config=OperatorConfig(
            ray_config=RayOperatorConfig(num_replicas=1),
        ),
    ).transform(
        Window(window_size=FLAGS.window_size, slide_size=FLAGS.slide_size),
        scheduler=FIFO(),
        operator_config=OperatorConfig(
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers),
        ),
    ).transform(
        STLFitForecast(FLAGS.results_dir),
        scheduler=schedulers[FLAGS.scheduler],
        operator_config=OperatorConfig(
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers),
        ),
    )

    app.deploy()
    app.wait()
    print("Finished", FLAGS.results_dir)


if __name__ == "__main__":
    app.run(main)
