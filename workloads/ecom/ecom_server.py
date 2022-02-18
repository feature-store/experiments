from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
from typing import List
import os
import time
from collections import defaultdict
import pandas as pd
import simpy
from absl import app, flags
import numpy as np
import torch
# import wandb

from model.xlnet import create_backbone

# might need to do  export PYTHONPATH='.'
# from workloads.util import read_config, use_dataset, log_dataset, log_results, WriteFeatures

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
    help="Scheduling policy (either FIFO or LIFO)",
    required=True,
)

flags.DEFINE_integer(
    "workers",
    default=2,
    help="Number of workers for bottlenck operator",
    required=False,
)

flags.DEFINE_string(
    "data_dir",
    default=None,
    help="Benchmark directory",
    required=True,
)

flags.DEFINE_string(
    "ckpt_dir",
    default=None,
    help="Checkpoint directory",
    required=True,
)


@dataclass
class Click:
    user_id: str
    item_id: int
    timestamp: int
    ingest_time: float


@dataclass
class UserHistory:
    user_id: str
    item_ids: List[int]
    timestamp: int
    ingest_time: float


@dataclass
class UserEmbedding:
    user_id: str
    item_ids: List[int]
    user_embedding: np.array
    timestamp: int
    ingest_time: float
    processing_time: float
    runtime: float


class ClickSource(BaseTransform):
    def __init__(self, data_dir: str) -> None:

        events_df = pd.read_csv(f"{data_dir}/events.csv")

        self.ts = 0
        self.data = events_df
        self.total = len(events_df.index)

    def on_event(self, _: Record) -> List[Record[Click]]:
        events = self.data[self.data["ts"] == self.ts].to_dict("records")

        num_remaining = len(self.data[self.data["ts"] >= self.ts].index)
        if num_remaining == 0:
            raise StopIteration()
        else: 
            print(f"Completed {num_remaining} / {self.total} ({(self.total - num_remaining) * 100/ self.total:.2f}%)")
        ingest_time = time.time()
        #if len(events) > 0:
        #    print("sending events", self.ts, len(events), "remaining", num_remaining)
        self.ts += 1
        return [
            Record(
                Click(
                    user_id=e["key_id"],
                    item_id=e["value"],
                    timestamp=e["ts"],
                    ingest_time=ingest_time,
                )
            )
            for e in events
        ]


class GroupByUser(BaseTransform):
    def __init__(self) -> None:
        self._data = defaultdict(list)

    def on_event(self, record: Record) -> Record[UserHistory]:
        self._data[record.entry.user_id].append(record.entry.item_id)

        return Record(
            UserHistory(
                user_id=record.entry.user_id,
                item_ids=self._data[record.entry.user_id],
                timestamp=record.entry.timestamp,
                ingest_time=record.entry.ingest_time,
            )
        )


class ComputeEmbedding(BaseTransform):
    def __init__(
        self,
        ckpt_dir: str,
    ):
        self.mask_index = 1 # NOTE: arbitrary non-zero index
        schema_file = os.path.join(ckpt_dir, "schema.pbtxt")
        self.model = create_backbone(schema_file, ckpt_dir)
        self.model.eval()

    @torch.no_grad()
    def on_event(self, record: Record) -> Record[UserEmbedding]:
        start = time.time()

        # NOTE: the batch size should be always 1
        input_item_ids = record.entry.item_ids + [self.mask_index]
        inputs = {
            "sess_pid_seq": torch.LongTensor(input_item_ids).view(1, -1),
        }
        user_embedding = self.model(inputs, training=False)[0, -1, :]
        user_embedding = user_embedding.numpy()

        end = time.time()

        return Record(
            UserEmbedding(
                user_id=record.entry.user_id,
                item_ids=record.entry.item_ids,
                user_embedding=user_embedding,
                timestamp=record.entry.timestamp,
                ingest_time=record.entry.ingest_time,
                processing_time=end,
                runtime=end - start,
            )
        )


class WriteFeatures(BaseTransform): 

    def __init__(self, filename: str):
        self.filename = filename
        self.logs: List[dict] = []

    def on_event(self, record: Record) -> None:
        ts = record.entry.timestamp + (record.entry.processing_time - record.entry.ingest_time)
        self.logs.append({
            "key_id": record.entry.user_id,
            "timestamp": ts,
            "feature": record.entry.user_embedding,
        })

    def __del__(self):
        df = pd.DataFrame(self.logs)
        df.to_parquet(self.filename)


def main(argv):
    print("Running Ecom pipeline on ralf...")
    name = f"results_workers_{FLAGS.workers}_{FLAGS.scheduler}"

    # data_dir = use_dataset(FLAGS.experiment, redownload=False)
    # results_dir = os.path.join(read_config()["results_dir"], FLAGS.experiment)
    # print("Using data from", data_dir)
    # print("Making results for", results_dir)

    # FIXME
    results_dir = "results"

    # create results file/directory
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_file = f"{results_dir}/{name}.csv"
    print("results file", results_file)

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
    xlnet4rec_ff = app.source(
        ClickSource(FLAGS.data_dir),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env,
                processing_time_s=0.01,
                stop_after_s=10,
            ),
            ray_config=RayOperatorConfig(num_replicas=1),
        ),
    ).transform(
        GroupByUser(),
        scheduler=FIFO(),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env,
                processing_time_s=0.01,
            ),
            ray_config=RayOperatorConfig(num_replicas=1),
        ),
    ).transform(
        ComputeEmbedding(FLAGS.ckpt_dir),
        scheduler=schedulers[FLAGS.scheduler],
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.2, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=FLAGS.workers)
        )
    ).transform(
        WriteFeatures(results_file),
    )

    app.deploy()
    app.wait()


if __name__ == "__main__":
    app.run(main)
