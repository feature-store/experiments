import argparse
from redis import Redis
import pickle
from pathlib import Path
import json
import os
import time
from typing import Dict, List

from absl import flags, app

import ray
import redis
from ralf.core import Ralf
from ralf.operator import DEFAULT_STATE_CACHE_SIZE, Operator
from ralf.operators.source import Source
from ralf.policies import load_shedding_policy, processing_policy
from ralf.state import Record, Schema
from ralf.table import Table
from statsmodels.tsa.seasonal import STL


SEASONALITY = 24 * 7

FLAGS = flags.FLAGS
flags.DEFINE_integer("window_size", 672, "window size for stl trainer")
flags.DEFINE_integer("global_slide_size", 48, "static slide size configuration")
flags.DEFINE_string(
    "per_key_slide_size_plan_path", "", "pre key slide size configuration"
)
flags.DEFINE_bool(
    "use_per_key_slide_size_plan", False, "whether or not to use the per key plan"
)
flags.DEFINE_integer("seasonality", SEASONALITY, "seasonality for stl training process")
flags.DEFINE_integer(
    "redis_model_db_id", default=2, help="Redis DB number for db snapshotting."
)

flags.DEFINE_string(
    "experiment_dir", None, "directory to write metadata to", required=True
)
flags.DEFINE_string("experiment_id", "", "experiment run name for wandb")
flags.DEFINE_bool("log_wandb", False, "whether to use wandb logging")

flags.DEFINE_integer("source_num_replicas", 1, "number of shard for source operator")
flags.DEFINE_integer("window_num_replicas", 1, "number of shard for window operator")
flags.DEFINE_integer("map_num_replicas", 1, "number of shard for map operator")
flags.DEFINE_integer("sink_num_replicas", 1, "number of shard for sink operator")


@ray.remote
class RedisSource(Source):
    def __init__(
        self,
        num_worker_threads: int = 1,
    ):
        schema = Schema(
            "key",
            {
                "key": str,
                "value": float,
                "timestamp": int,
                "send_time": float,
                "create_time": float,
            },
        )
        super().__init__(schema, num_worker_threads=num_worker_threads)
        self.consumer = redis.Redis()

    def next(self) -> List[Record]:
        _, data = self.consumer.xreadgroup(
            "ralf-reader-group",
            f"reader-{self._shard_idx}",
            {"ralf": ">"},
            count=1,
            block=100 * 1000,
        )[0]
        record_id, payload = data[0]
        self.last_record = record_id
        self.consumer.xack("ralf", "ralf-reader-group", record_id)

        assert isinstance(payload, dict)
        record = Record(
            key=payload[b"key"].decode(),
            value=payload[b"value"].decode(),
            timestamp=int(payload[b"timestamp"]),
            send_time=float(payload[b"send_time"]),
            create_time=time.time(),
        )
        return [record]


@ray.remote
class STLTrainer(Operator):
    def __init__(
        self,
        seasonality,
        num_worker_threads=1,
        processing_policy=processing_policy.last_completed,
        load_shedding_policy=load_shedding_policy.later_complete_time,
    ):

        schema = Schema(
            "key",
            {
                "key": str,
                "trend": float,
                "seasonality": List[float],
                "send_time": float,
                "create_time": float,
                "complete_time": float,
                "timestamp": int,
            },
        )
        super().__init__(
            schema,
            num_worker_threads=num_worker_threads,
            processing_policy=processing_policy,
            load_shedding_policy=load_shedding_policy,
        )
        self.robust = True
        self.period = seasonality

    def on_record(self, record: Record) -> Record:
        values = [r.value for r in record.window]
        stl_result = STL(values, period=self.period, robust=self.robust).fit()

        return Record(
            key=record.key,
            trend=stl_result.trend[-1],
            seasonality=list(stl_result.seasonal[-(self.period + 1) : -1]),
            create_time=record.create_time,
            complete_time=time.time(),
            timestamp=record.timestamp,
        )


@ray.remote
class RedisSink(Operator):
    def __init__(self, db_id: int, num_worker_threads: int = 1):
        super().__init__(
            schema=None,
            num_worker_threads=num_worker_threads,
        )
        self.redis_conn = Redis(db=db_id)

    def on_record(self, record: Record):
        self.redis_conn.set(record.key, pickle.dumps(record.entries))


def create_stl_pipeline(metrics_dir):

    # create Ralf instance
    ralf_conn = Ralf(
        metric_dir=os.path.join(FLAGS.experiment_dir, FLAGS.experiment_id),
        log_wandb=FLAGS.log_wandb,
        exp_id=FLAGS.experiment_id,
    )

    source = Table(
        [],
        RedisSource,
        num_replicas=FLAGS.source_num_replicas,
        num_worker_threads=1,
    )

    (
        source.window(
            FLAGS.window_size,
            FLAGS.global_slide_size,
            num_replicas=FLAGS.window_num_replicas,
            num_worker_threads=1,
            per_key_slide_size_plan_file=FLAGS.per_key_slide_size_plan_path
            if FLAGS.use_per_key_slide_size_plan
            else None,
        )
        .map(
            STLTrainer,
            FLAGS.seasonality,
            num_replicas=FLAGS.map_num_replicas,
            num_worker_threads=1,
        )
        .map(
            RedisSink,
            num_replicas=FLAGS.sink_num_replicas,
            num_worker_threads=1,
            db_id=FLAGS.redis_model_db_id,
        )
    )

    # TODO(simon): user should be able to deploy pipeline as a whole, just deploying source is hard to use.
    # deploy
    ralf_conn.deploy(source, "source")

    return ralf_conn


def _get_config() -> Dict:
    """Return all the flag vlaue defined here."""
    return {f.name: f.value for f in FLAGS.get_flags_for_module("__main__")}


def _perform_periodic_snapshot(
    ralf_conn: Ralf, *, run_duration_s: int, snapshot_interval: int
):
    start = time.time()
    while time.time() - start < run_duration_s:
        snapshot_time = ralf_conn.snapshot()
        remaining_time = snapshot_interval - snapshot_time
        if remaining_time < 0:
            print(
                f"snapshot interval is {snapshot_interval} but it took {snapshot_time} to perform it!"
            )
            time.sleep(0)
        else:
            print("writing snapshot", snapshot_time)
            time.sleep(remaining_time)
    print("TIME COMPLETE", run_duration_s)


def main(argv):
    print("Using config", _get_config())

    # create experiment directory.
    exp_dir = Path(FLAGS.experiment_dir) / FLAGS.experiment_id
    exp_dir.mkdir(exist_ok=True, parents=True)
    with open(exp_dir / "server_config.json", "w") as f:
        json.dump(_get_config(), f)

    # create stl pipeline.
    ralf_conn = create_stl_pipeline(exp_dir / "metrics")
    ralf_conn.run()

    # Keep it running with system snapshot.
    _perform_periodic_snapshot(ralf_conn, run_duration_s=6000, snapshot_interval=10)


if __name__ == "__main__":
    app.run(main)
