import json
from typing import DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from functools import cmp_to_key

import configparser

import pandas as pd

import simpy
from ralf.state import Record
from ralf.policies.load_shedding_policy import (
    always_process,
    make_mean_policy,
    make_sampling_policy,
)
from ralf.policies.processing_policy import fifo, lifo  # , make_sorter_with_key_weights
from ralf.simulation.priority_queue import PerKeyPriorityQueue
from ralf.simulation.source import JSONSource
from ralf.simulation.window import WindowOperator
from ralf.simulation.mapper import RalfMapper, RoundRobinLoadBalancer


from ralf.policies.load_shedding_policy import (
    always_process,
    newer_processing_time,
    later_complete_time,
    make_sampling_policy,
    make_mean_policy,
    make_cosine_policy,
)


class WikiMapper(RalfMapper):

    # def __init__(
    #    self,
    #    env: simpy.Environment,
    #    source_queues: Dict[KeyType, PerKeyPriorityQueue],
    #    key_selection_policy_cls: Type[CrossKeyLoadBalancer],
    #    model_run_time_s: float,
    # ) -> None:

    #    self.env = env
    #    self.source_queues = source_queues
    #    self.key_selection_policy = key_selection_policy_cls()
    #    self.model_runtime_s = model_run_time_s
    #    self.env.process(self.run())

    #    self.ready_time_to_batch: Dict[float, List[Tuple[int, float]]] = {}

    def run(self):
        while True:
            # windows = yield self.source_queue.get()
            chosen_key = self.key_selection_policy.choose(self.source_queues)
            windows = yield self.source_queues[chosen_key].get()
            print(
                f"at time {self.env.now:.2f}, RalfMapper should work on {windows} (last timestamp)"
            )
            edits = [(val, chosen_key) for val in windows.window[0].value]
            print("edits", edits)

            if self.env.now in self.ready_time_to_batch:
                self.ready_time_to_batch[self.env.now] += edits
            else:
                self.ready_time_to_batch[self.env.now] = edits

            yield self.env.timeout(self.model_runtime_s)


policies = {
    "fifo": fifo,
    "lifo": lifo,
    "always_process": always_process,
    "sample_half": make_sampling_policy(0.5),
}


def run_once(
    out_path: str,
    prioritization_policy: str,
    load_sheeding_policy: str,
    keys: List[str],
    per_key_records_per_second: int,
    total_runtime_s: float,
    model_runtime_constant: float,
    data_file: str = None,
):

    env = simpy.Environment()

    source_to_window_queue = simpy.Store(env)
    windows_to_mapper_queue = {
        key: PerKeyPriorityQueue(
            env,
            processing_policy=policies[prioritization_policy],
            load_shedding_policy=policies[load_sheeding_policy],
        )
        for key in keys
    }

    JSONSource(
        env,
        records_per_sec_per_key=per_key_records_per_second,
        num_keys=len(keys),
        next_queue=source_to_window_queue,
        total_run_time=total_runtime_s,
        data_file=data_file,
    )

    WindowOperator(
        env,
        window_size=1,
        slide_size=1,
        source_queue=source_to_window_queue,
        next_queues=windows_to_mapper_queue,
    )

    m = WikiMapper(
        env,
        source_queues=windows_to_mapper_queue,
        model_run_time_s=model_runtime_constant,
        key_selection_policy_cls=RoundRobinLoadBalancer,
    )
    env.run(until=total_runtime_s)

    plan = m.ready_time_to_batch
    with open(out_path, "w") as f:
        json.dump(plan, f)


if __name__ == "__main__":

    # load sheding: random, drop short edits
    # prioritization: prioritize most recent version
    # cross-key prioritzation: historical page views,

    # configuration file
    config = configparser.ConfigParser()
    config.read("config.yml")
    plan_dir = config["simulation"]["plan_dir"]
    init_data_file = config["simulation"]["init_data_file"]
    stream_edits_file = config["simulation"]["stream_edits_file"]
    stream_questions_file = config["simulation"]["stream_questions_file"]

    # load simulation data
    edits = json.load(open(stream_edits_file))
    init_data = json.load(open(init_data_file))
    keys = list(init_data.keys())

    # policies
    prioritization_policies = ["fifo", "lifo"]
    load_shedding_policies = ["always_process"]
    model_runtimes = [0.000001, 0.00001, 0.0000001, 0.000000001, 0]
    records_per_second = [100]

    for prio_policy in prioritization_policies:
        for load_shed_policy in load_shedding_policies:
            for runtime in model_runtimes:
                for rate in records_per_second:

                    out_path = f"{plan_dir}/plan-{prio_policy}-{load_shed_policy}-{runtime}-{rate}.json"
                    print("running", out_path, runtime)
                    run_once(
                        out_path,
                        prio_policy,
                        load_shed_policy,
                        keys,
                        per_key_records_per_second=rate,
                        total_runtime_s=len(edits),
                        model_runtime_constant=runtime,
                        data_file=stream_edits_file,
                    )
                    print("done", out_path)
