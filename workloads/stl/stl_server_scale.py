import asyncio
import json
import os
import pickle
import threading
import time
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np
import pandas as pd
import ray
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
from sktime.performance_metrics.forecasting import mean_squared_scaled_error
from sortedcontainers import SortedSet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from tqdm import tqdm

logger = get_logger()
FLAGS = flags.FLAGS
WINDOW_SIZE = 864
TOTAL_LENGTH = 8639
SLIDE_SIZE = 288
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
        feature: Optional[Record] = self._operator.get(record.entry.key_id)
        if feature is None:
            # logger.msg(
            #     f"Missing feature for key {record.entry.shard_key}, returning max_prio"
            # )
            return self.max_prio
        return self.compute_priority(record)

    def push_event(self, record: Record):
        if record.is_stop_iteration():
            self.stop_iteration = record
            self.wake_waiter_if_needed()
            return

        record_key = record.entry.key_id
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
        return self.key_to_priority.get(record.entry.key_id, 0) + 1


class CumulativeErrorScheduler(BasePriorityScheduler):
    """Prioritize the key that has highest prediction error so far"""

    def __init__(self):
        super().__init__()

    def compute_priority(self, record: Record["RealWindowValue"]) -> float:
        assert isinstance(record.entry, RealWindowValue)

        # lookup current feature
        feature: Record[TimeSeriesValue] = self._operator.get(record.entry.key_id)
        y_true = record.entry.values
        y_pred = feature.y_pred[
            record.entry.start_window_idx : record.entry.start_window_idx + WINDOW_SIZE
        ]
        y_train = feature.y_train  # latest window used to generate the prediction

        def allna(arr):
            return np.isnan(arr).sum() == len(arr)

        if allna(y_pred):
            return self.max_prio

        def fillna(arr):
            assert not allna(arr), "arr shouldn't be full of nans"
            return np.nan_to_num(arr, nan=np.nanmean(arr))

        error = mean_squared_scaled_error(
            y_true=fillna(y_true),
            y_pred=fillna(y_pred),
            y_train=fillna(y_train),
        )

        return error


class AzureDataSource:
    def __init__(self, key_to_shard_group_map) -> None:
        self.key_to_shard_group_map = key_to_shard_group_map

        self.conn = duckdb.connect(
            "/home/ubuntu/azure_long_series_with_ts_array.duckdb", read_only=True
        )
        self.conn.execute("pragma enable_progress_bar").fetchall()

        self.arrival_df: pd.DataFrame = None
        self.window_start_idx_range: Tuple[int, int] = None
        self.ts_array_df: pd.DataFrame = None

        self.load_arrival_df()
        self.load_ts_array_df()

    def load_arrival_df(self):
        cache_path = "/home/ubuntu/arrival_df.cache.parquet"

        if os.path.exists(cache_path):
            arrival_df = pd.read_parquet(cache_path)
        else:
            self.conn.execute("select setseed(0.42)")
            cursor = self.conn.execute(
                """
            select window_start_idx, count(*), list(int_id), list(original_window_start_idx)
            from (
                select int_id, window_start_idx as original_window_start_idx, window_start_idx + start_offset as window_start_idx
                from (
                    select int_id, unnest(range(timestamp[1], timestamp[-1]+1-864, 288)) as window_start_idx, start_offset
                    from (
                        select int_id, timestamp, round(random()*288, 0) as start_offset
                        from readings
                    )
                )
            )
            group by window_start_idx
            order by window_start_idx;
            """
            )
            arrival_df: pd.DataFrame = (
                cursor.fetch_record_batch().read_next_batch().to_pandas()
            )
            start_idx_col = arrival_df["window_start_idx"].astype("int")
            arrival_df["window_start_idx"] = start_idx_col
            arrival_df = arrival_df.set_index("window_start_idx")
            arrival_df.to_parquet(cache_path)

        start_idx_col = arrival_df.index
        self.window_start_idx_range = (start_idx_col.min(), start_idx_col.max())
        self.arrival_df = arrival_df

    def load_ts_array_df(self):
        cache_path = "/home/ubuntu/ts_array_df.cache.parquet"
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
        else:
            df = (
                self.conn.execute(
                    """
                    SELECT int_id, ts_array
                    FROM readings
                    """
                ).df()  # takes 10min. However, the calls below take 3min but inaccurate.
                # .fetch_record_batch()
                # .read_next_batch()
                # .to_pandas()
            )
            df = df.set_index("int_id")
            df.to_parquet(cache_path)
        self.ts_array_df = df

    def read_real_input_for_round(self, round_idx: int):
        if not (
            self.window_start_idx_range[0]
            <= round_idx
            <= self.window_start_idx_range[1]
        ):
            raise StopIteration()

        # Later in the time series, there might be a time when no new windows need to be sent.
        if round_idx not in self.arrival_df.index:
            return []

        plan_series = self.arrival_df.loc[round_idx]
        assert isinstance(plan_series, pd.Series), (plan_series, round_idx)

        records = []
        for int_id, start_idx in zip(
            plan_series["list(int_id)"], plan_series["list(original_window_start_idx)"]
        ):
            record = Record(
                entry=RealWindowValue(
                    key_id=int_id,
                    values=np.array(self.ts_array_df.loc[int_id, "ts_array"])[
                        start_idx : start_idx + WINDOW_SIZE
                    ],
                    start_window_idx=start_idx,
                    shard_key=self.key_to_shard_group_map[int_id],
                )
            )
            if np.isnan(record.entry.values).sum() == WINDOW_SIZE:
                print(f"warning: input all nans for {int_id}, {start_idx}")
            else:
                records.append(record)
        return records


@dataclass
class RealWindowValue:
    key_id: int
    values: List[float]
    start_window_idx: int
    shard_key: int


class DataSource(BaseTransform):
    """Generate event data over keys"""

    def __init__(
        self,
        sleep: float,
        start_signal: ray.ObjectRef,
        key_to_shard_group_map,
        results_dir: str,
    ):
        self.sleep = sleep
        self.start_signal = start_signal
        self.key_to_shard_group_map = key_to_shard_group_map
        self.results_dir = results_dir

    def prepare(self):
        os.makedirs(self.results_dir, exist_ok=True)
        self.send_time_log = open(
            os.path.join(self.results_dir, "DataSouce.log.jsonl"), "w"
        )
        self.azure_source = AzureDataSource(self.key_to_shard_group_map)
        self.ts = 0
        ray.get(self.start_signal)

    def on_shutdown(self):
        self.send_time_log.flush()

    def on_event(self, _: Record) -> List[Record[RealWindowValue]]:
        print(f"sending {self.ts}")
        batch = self.azure_source.read_real_input_for_round(self.ts)

        self.send_time_log.write(
            json.dumps(
                {"current_time": time.time(), "ts": self.ts, "batch_length": len(batch)}
            )
        )
        self.send_time_log.write("\n")

        self.ts += 1
        if self.sleep > 0:
            time.sleep(self.sleep)
        return batch


@dataclass
class TimeSeriesValue:
    key_id: int
    y_pred: np.ndarray
    y_train: np.ndarray


class ForecastArrayManager:
    def __init__(self, total_workers, shard_idx, key_to_shard_group_map) -> None:
        self.idx_df = pd.read_parquet("/home/ubuntu/start_end_idx_per_id.parquet")

        self.key_to_shard_group_map = key_to_shard_group_map
        shards_responsible_for = set()
        total_shards = 800
        for shard in range(total_shards):
            if shard % total_workers == shard_idx:
                shards_responsible_for.add(shard)

        keys = set()
        for key, shard_idx in key_to_shard_group_map.items():
            if shard_idx in shards_responsible_for:
                keys.add(key)

        self.forecasts: Dict[KeyType, np.ndarray] = dict()
        for key in keys:
            pred_arr = np.empty(TOTAL_LENGTH, dtype="float32")
            pred_arr.fill(np.nan)
            self.forecasts[key] = pred_arr

        self.pre_generated_windows_shard_to_df: Dict[int, pd.DataFrame] = {}
        for shard_idx in shards_responsible_for:
            self.pre_generated_windows_shard_to_df[shard_idx] = pd.read_parquet(
                f"/ssd_data/windows-compact-v3/{shard_idx}.parquet"
            )

        self._key_already_prefilled_default: Set[KeyType] = set()

    def _get_pregenerated_windows(self, key):
        shard_idx = self.key_to_shard_group_map[key]
        df = self.pre_generated_windows_shard_to_df[shard_idx]
        return df[df["int_id"] == key]

    def fill_prediction(
        self,
        key: KeyType,
        window_start_idx: int,
        real_forecast_value: Optional[np.ndarray] = None,
    ):
        df = self._get_pregenerated_windows(key)
        pred_arr = self.forecasts[key]

        int_id = key
        range_series = self.idx_df.loc[int_id]
        ground_truth_start_idx, ground_truth_end_idx = (
            range_series["start_idx"],
            range_series["end_idx"],
        )

        relative_window_start_idx = window_start_idx - ground_truth_start_idx

        def fill_into_array(start_idx, forecase_value):
            pred_start_idx = ground_truth_start_idx + start_idx + WINDOW_SIZE
            remaining_pred_size = ground_truth_end_idx - pred_start_idx
            if remaining_pred_size <= 0:
                return
            forecast_to_fill = forecase_value[: remaining_pred_size + 1]
            pred_arr[
                pred_start_idx : pred_start_idx + remaining_pred_size + 1
            ] = forecast_to_fill

        if key not in self._key_already_prefilled_default:
            # We would want to default predictions to some herustics
            # for all windows before this predictions.
            for _, row in df.iterrows():
                if row["start_idx"] < relative_window_start_idx:
                    fill_into_array(
                        row["start_idx"],
                        np.ones_like(row["forecast_arr"]) * row["window_arr"].mean(),
                    )
                    self._key_already_prefilled_default.add(key)
                else:
                    break

        # Perform the filling for this prediction
        row = df[df["start_idx"] == relative_window_start_idx]
        assert len(row) == 1, (relative_window_start_idx, key, window_start_idx)
        row = row.squeeze()
        if real_forecast_value is not None:
            fill_into_array(row["start_idx"], real_forecast_value)
        else:
            fill_into_array(row["start_idx"], row["forecast_arr"])

    def get_y_pred_array(self, key: KeyType):
        return self.forecasts[key]


class STLFitForecast(BaseTransform):
    """
    Fit STLForecast model and forecast future points.
    """

    def __init__(
        self, total_num_workers: int, barrier_actor, key_to_shard_group_map, results_dir
    ):
        self.num_updates = 0

        self.total_num_workers = total_num_workers
        self.barrier_actor = barrier_actor
        self.key_to_shard_group_map = key_to_shard_group_map

        self.results_dir = results_dir

    def prepare(self):
        os.makedirs(self.results_dir, exist_ok=True)

        # self.start_time = time.time()
        self.data = defaultdict(lambda: None)

        self.provider = ForecastArrayManager(
            self.total_num_workers,
            self.operator_context["shard_idx"],
            self.key_to_shard_group_map,
        )
        self.barrier_actor.worker_ready.remote()
        self.log_file = open(
            os.path.join(
                self.results_dir, f"{self.operator_context['shard_idx']}.log.jsonl"
            ),
            "w",
        )

    def on_shutdown(self):
        self.log_file.flush()
        with open(
            os.path.join(
                self.results_dir, f"{self.operator_context['shard_idx']}.forecasts.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(self.provider.forecasts, f)

    def get(self, key):
        return self.data[key]

    def on_event(self, record: Record[RealWindowValue]):
        key_id = record.entry.key_id

        # if np.isnan(record.entry.values).sum() == len(record.entry.values):
        #     raise Exception(
        #         f"{key_id}'s y_train is all nans. {record.entry.start_window_idx}, {record.entry.values}"
        #     )

        with warnings.catch_warnings():
            # catch warning for ML fit
            warnings.filterwarnings("ignore")
            model = STLForecast(
                pd.Series(record.entry.values).interpolate(),
                ARIMA,
                model_kwargs=dict(order=(1, 1, 0), trend="t"),
                period=12 * 24,  # 5 min timestamp interval, period of one day
            ).fit()
            forecast = model.forecast(7775).values

        self.log_file.write(
            json.dumps(
                {
                    "key_id": int(key_id),
                    "start_window_idx": int(record.entry.start_window_idx),
                    "current_time": time.time(),
                }
            )
        )
        self.log_file.write("\n")

        # self.num_updates += 1
        # if self.num_updates % 1000:
        #     print(
        #         "avg throughput",
        #         self.num_updates,
        #         self.num_updates / (time.time() - self.start_time),
        #     )

        self.provider.fill_prediction(key_id, record.entry.start_window_idx, forecast)
        y_pred_array = self.provider.get_y_pred_array(key_id)

        forecast_record = TimeSeriesValue(
            key_id=key_id,
            y_pred=y_pred_array,
            y_train=record.entry.values,
        )

        self.data[key_id] = forecast_record


def _get_config() -> Dict:
    """Return all the flag vlaue defined here."""
    return {f.name: f.value for f in FLAGS.get_flags_for_module("__main__")}


flags.DEFINE_string(
    "scheduler",
    default=None,
    help="Scheduling policy for STL operator",
    required=True,
)
flags.DEFINE_integer(
    "workers",
    default=None,
    help="Number of workers for bottlenck operator",
    required=True,
)
flags.DEFINE_string(
    "results_dir",
    default=f"results/stl/results/{int(time.time())}",
    help="Diretory to write result jsonl to",
    required=False,
)
flags.DEFINE_float(
    "source_sleep_per_batch",
    default=None,
    required=True,
    help="source sleep duration in s",
)


@ray.remote
class Barrier:
    def __init__(self, num):
        self.num = num
        self.ready_num = 0

        self.event = asyncio.Event()

    def worker_ready(self):
        self.ready_num += 1
        print(f"Signal received {self.ready_num}/{self.num}")
        if self.ready_num == self.num:
            self.event.set()

    async def source_can_go(self):
        await self.event.wait()


def main(argv):
    logger.msg("Running STL pipeline on ralf...")

    os.makedirs(FLAGS.results_dir)
    logger.msg(f"Results dir {FLAGS.results_dir}")
    with open(f"{FLAGS.results_dir}/config.json", "w") as f:
        json.dump(_get_config(), f)

    os.makedirs(f"{FLAGS.results_dir}/metrics")
    app = RalfApplication(
        RalfConfig(deploy_mode="ray", metrics_dir=f"{FLAGS.results_dir}/metrics")
    )

    with open("/home/ubuntu/windows-compact-v3/file-to-keys.json") as f:
        key_to_shard_group_map = {}
        for k, v in json.load(f).items():
            for key_id in v:
                key_to_shard_group_map[key_id] = int(k)

    # scheduler options
    schedulers = {
        "lifo": KeyAwareLifo(),
        "rr": RoundRobinScheduler(),
        "ce": CumulativeErrorScheduler(),
    }

    barrier_actor = Barrier.options(num_cpus=0).remote(FLAGS.workers)

    app.source(
        DataSource(
            sleep=FLAGS.source_sleep_per_batch,
            start_signal=barrier_actor.source_can_go.remote(),
            key_to_shard_group_map=key_to_shard_group_map,
            results_dir=FLAGS.results_dir,
        ),
        operator_config=OperatorConfig(
            ray_config=RayOperatorConfig(
                num_replicas=1,
                actor_options={
                    "num_cpus": 0,
                    "resources": {"node:172.31.62.177": 0.01},
                },
            ),
        ),
    ).transform(
        STLFitForecast(
            total_num_workers=FLAGS.workers,
            barrier_actor=barrier_actor,
            key_to_shard_group_map=key_to_shard_group_map,
            results_dir=FLAGS.results_dir,
        ),
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
