import json
from typing import DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from functools import cmp_to_key

import configparser

import pandas as pd

import simpy
from ralf.state import Record
from ralf.load_shedding_policy import (
    always_process,
    make_mean_policy,
    make_sampling_policy,
)
from ralf.processing_policy import fifo, lifo  # , make_sorter_with_key_weights
from ralf.simulation.priority_queue import PerKeyPriorityQueue
from ralf.simulation.source import WikiSource
from ralf.simulation.mapper import WikiMapper, RoundRobinLoadBalancer


from ralf.load_shedding_policy import (
    always_process,
    newer_processing_time,
    later_complete_time,
    make_sampling_policy,
    make_mean_policy,
    make_cosine_policy,
)

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

    mapper_queue = {
        key: PerKeyPriorityQueue(
            env,
            processing_policy=policies[prioritization_policy],
            load_shedding_policy=policies[load_sheeding_policy],
        )
        for key in keys
    }

    s = WikiSource(
        env,
        records_per_sec_per_key=per_key_records_per_second,
        num_keys=len(keys),
        next_queues=mapper_queue,
        total_run_time=total_runtime_s,
        data_file=data_file,
    )
    m = WikiMapper(
        env,
        source_queues=mapper_queue,
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
    plan_dir = config['simulation']['plan_dir']
    init_data_file = config['simulation']['init_data_file']
    stream_edits_file = config['simulation']['stream_edits_file']
    stream_questions_file = config['simulation']['stream_questions_file']

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
