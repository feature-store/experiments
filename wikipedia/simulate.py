import json 
import itertools
from typing import DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from functools import cmp_to_key
import random

import configparser
import argparse

import pandas as pd

import wandb

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
from ralf.simulation.mapper import (
    RalfMapper,
    RoundRobinLoadBalancer,
    CrossKeyLoadBalancer,
)


from ralf.policies.load_shedding_policy import (
    always_process,
    newer_processing_time,
    later_complete_time,
    make_sampling_policy,
    make_mean_policy,
    make_cosine_policy,
)

from typing import Dict, List, Tuple, Type

from preprocessing.log_data import log_plans

def current_weights(ts, ts_to_weights): 
    ts = int(ts)
    min_dist = max(list(ts_to_weights.keys()))

    index = 0
    for key in ts_to_weights.keys():
        if key >= ts:
            break
        index = key

    return ts_to_weights[key]

class RoundRobinLoadBalancerFix(CrossKeyLoadBalancer):
    """Simple policy that cycle through all the keys fairly"""

    def __init__(self):
        self.cur_key_set = set()
        self.cur_key_iter = None

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:
        key_set = set(per_key_queues.keys())
        if key_set != self.cur_key_set:
            self.cur_key_set = key_set
            self.cur_key_iter = itertools.cycle(key_set)

        key = next(self.cur_key_iter)
        while per_key_queues[key].size() == 0:
            key = next(self.cur_key_iter)
        # TODO(simon): maybe do a "peak" here to trigger eviction policies
        return key


class WeightedRoundRobin(CrossKeyLoadBalancer):
    """Simple policy that cycle through all the keys fairly"""

    def __init__(self, pageview_file, all_keys):
        self.cur_key_set = []
        self.cur_key_iter = None
        pageview_df = pd.read_csv(pageview_file)

        self.weights = json.load(open("weights.json"))

        ##self.raw_weights = pageview_df.set_index("doc_id")["weights"].to_dict()
        #self.raw_weights = pageview_df.set_index("doc_id")["2021090300"].to_dict()
        #self.weights = {}
        #for key in self.raw_weights.keys(): 
        #    if str(key) not in all_keys: 
        #        continue 

        #    self.weights[key] = int(self.raw_weights[key]*1000)
        #    #assert self.weights[key] > 0, f"Too small {key}, {self.raw_weights[key]}"
        #    if self.weights[key] == 0:
        #        self.weights[key] = 1


        for key in all_keys: 
            if key not in self.weights: 
                self.weights[key] = 1


        for key in self.weights.keys(): 
            for i in range(self.weights[key]):
                self.cur_key_set.append(str(key))
        random.shuffle(self.cur_key_set)
        self.cur_key_iter = itertools.cycle(self.cur_key_set)


    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:

        key = next(self.cur_key_iter)
        while per_key_queues[key].size() == 0:
            key = next(self.cur_key_iter)
        # TODO(simon): maybe do a "peak" here to trigger eviction policies
        return key

class AdaptiveWeightedRoundRobin(CrossKeyLoadBalancer):
    """Simple policy that cycle through all the keys fairly"""

    def __init__(self, timestamp_weights_file):
        self.cur_key_set = []
        self.cur_key_iter = None

        pageview_df = pd.read_csv(pageview_file)
        self.raw_weights = pageview_df.set_index("doc_id")["weights"].to_dict()
        self.weights = {}
        for key in self.raw_weights.keys(): 
            if str(key) not in all_keys: 
                continue 

            self.weights[key] = int(self.raw_weights[key]*1000)
            assert self.weights[key] > 0, f"Too small {key}, {self.raw_weights[key]}"


        for key in self.weights.keys(): 
            for i in range(self.weights[key]):
                self.cur_key_set.append(str(key))
        random.shuffle(self.cur_key_set)
        self.cur_key_iter = itertools.cycle(self.cur_key_set)


    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:

        key = next(self.cur_key_iter)
        while per_key_queues[key].size() == 0:
            key = next(self.cur_key_iter)
        # TODO(simon): maybe do a "peak" here to trigger eviction policies
        return key


class AdaptiveWeightedLoadBalancer(CrossKeyLoadBalancer):

    def __init__(self, timestamp_weights_file):
        data = json.load(open(timestamp_weights_file))
        self.timestamp_weights = {}
        for key in data.keys(): 
            self.timestamp_weights[int(key)] = data[key]

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], timestamp: int) -> str:
        weights_map = current_weights(timestamp, self.timestamp_weights)

        chosen_key = None
        max_len = 0
        total_len = 0
        keys = []
        weights = []
        for key in per_key_queues.keys():
            size = per_key_queues[key].size()
            if size >= 1 and key in weights_map:
                keys.append(key)
                weights.append(weights_map[key])
            total_len += size
        chosen_key = random.choices(keys, weights, k=1)[0]
        return chosen_key


class WeightedLoadBalancer(CrossKeyLoadBalancer):

    def __init__(self, pageview_file):
        pageview_df = pd.read_csv(pageview_file)
        #self.weights = pageview_df.set_index("doc_id")["weights"].to_dict()
        self.weights = json.load(open("weights.json"))

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:
        chosen_key = None
        max_len = 0
        total_len = 0
        keys = []
        weights = []
        for key in per_key_queues.keys():
            size = per_key_queues[key].size()
            if size >= 1 and int(key) in self.weights:
                keys.append(key)
                weights.append(self.weights[int(key)])
            total_len += size

        chosen_key = random.choices(keys, weights, k=1)[0]
        #print("choose", chosen_key, keys, weights)
        return chosen_key

class RandomLoadBalancer(CrossKeyLoadBalancer):

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:
        chosen_key = None
        max_len = 0
        total_len = 0
        keys = []
        for key in per_key_queues.keys():
            size = per_key_queues[key].size()
            if size >= 1:
                keys.append(key)
            total_len += size

        chosen_key = random.choices(keys, k=1)[0]
        return chosen_key


class WeightedLongestQueueLoadBalancer(CrossKeyLoadBalancer):

    def __init__(self, pageview_file):
        pageview_df = pd.read_csv(pageview_file)
        self.weights = pageview_df.set_index("doc_id")["weights"].to_dict()
        #print(self.weights)

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:
        chosen_key = None
        max_len = 0
        total_len = 0
        for key in per_key_queues.keys():
            size = per_key_queues[key].size()
            if int(key) not in self.weights:
                continue
            weighted_size = self.weights[int(key)]*self.weights[int(key)]
            if weighted_size > max_len:
                chosen_key = key
                max_len = size
            total_len += size
        #print(chosen_key, max_len, self.weights[int(chosen_key)])
        per_key_queues[chosen_key].clear()
        print("clear", chosen_key, total_len, per_key_queues[chosen_key].size())
        return chosen_key

class WeightedLoadBalancer(CrossKeyLoadBalancer):

    def __init__(self, pageview_file):
        pageview_df = pd.read_csv(pageview_file)
        self.weights = pageview_df.set_index("doc_id")["weights"].to_dict()
        #print(self.weights)

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:
        chosen_key = None
        max_len = 0
        total_len = 0
        keys = []
        weights = []
        for key in per_key_queues.keys():
            size = per_key_queues[key].size()
            if size >= 1 and int(key) in self.weights:
                keys.append(key)
                weights.append(self.weights[int(key)])
            total_len += size

        chosen_key = random.choices(keys, weights, k=1)[0]
        #print("choose", chosen_key, keys, weights)
        return chosen_key

class LongestQueueLoadBalancer(CrossKeyLoadBalancer):

    def choose(self, per_key_queues: Dict[str, PerKeyPriorityQueue], ts) -> str:
        chosen_key = None
        max_len = 0
        total_len = 0
        for key in per_key_queues.keys():
            size = per_key_queues[key].size()
            if size > max_len:
                chosen_key = key
                max_len = size
            total_len += size
        per_key_queues[chosen_key].clear()

        return chosen_key


class WikiMapper(RalfMapper):
    def __init__(
        self,
        env: simpy.Environment,
        source_queues: Dict[str, PerKeyPriorityQueue],
        key_selection_policy_cls: Type[CrossKeyLoadBalancer],
        model_run_time_s: float,
        keys: List[str],
    ) -> None:

        super().__init__(env, source_queues, key_selection_policy_cls, model_run_time_s)
        self.keys = keys
        self.source_queues = source_queues

        # self.env = env
        # self.source_queues = source_queues
        # self.key_selection_policy = key_selection_policy_cls()
        # self.model_runtime_s = model_run_time_s
        # self.env.process(self.run())

        self.ready_time_to_batch: Dict[float, List[Tuple[int, float]]] = {}

    def run(self, replica_id: int):

        self.source_queues = {
            key: self.total_source_queues[key] for key in self.sharded_keys[replica_id]
        }

        while True:
            yield simpy.AnyOf(self.env, [q.wait() for q in self.source_queues.values()])

            # choose key
            print("env time", self.env.now)
            chosen_key = self.key_selection_policy.choose(
                self.source_queues,
                self.env.now*100
            )
            assert chosen_key is not None

            # make sure queue size OK - jk doesn't work with dropping
            # assert total_size_orig == 0 or total_size == total_size_orig, f"Bad queue size {total_size_orig} -> {total_size}"

            # get chosen key
            windows = yield self.source_queues[chosen_key].get()
            # print(
            #     f"at time {self.env.now:.2f}, RalfMapper should work on {windows} (last timestamp), queue size {total_size}, wait time {self.model_runtime_s}"
            # )
            edits = [(val, windows.key) for val in windows.window[0].value]

            if self.env.now in self.ready_time_to_batch:
                self.ready_time_to_batch[self.env.now] += edits
            else:
                self.ready_time_to_batch[self.env.now] = edits

            yield self.env.timeout(self.model_runtime_s)


# configuration file
config = configparser.ConfigParser()
config.read("config.yml")
plan_dir = config["simulation"]["plan_dir"]
#init_data_file = config["simulation"]["init_data_file"]
#stream_edits_file = config["simulation"]["stream_edits_file"]
#stream_questions_file = config["simulation"]["stream_questions_file"]
#pageview_file = config["files"]["pageview_file"]
#timestamp_weights_file = config["files"]["timestamp_weights_file"]

run = wandb.init(job_type="dataset-creation", project="wiki-workload")
question_dir = run.use_artifact('ucb-ralf/wiki-workload /questions:v2', type='dataset').download()
simulation_dir = run.use_artifact('ucb-ralf/wiki-workload /simulation:v2', type='dataset').download()
pageview_dir = run.use_artifact('ucb-ralf/wiki-workload /pageviews:v0', type='dataset').download()

init_data_file = f"{simulation_dir}/init_data.json"
stream_edits_file = f"{simulation_dir}/edit_stream.json"
stream_questions_file = f"{simulation_dir}/question_stream.json"
pageview_file = f"{pageview_dir}/pageviews.csv"
timestamp_weights_file = f"{pageview_dir}/timestamp_weights_file.json"

# load simulation data
edits = json.load(open(stream_edits_file))
init_data = json.load(open(init_data_file))
keys = list(init_data.keys())

policies = {
    "fifo": fifo,
    "lifo": lifo,
    "always_process": always_process,
    "sample_half": make_sampling_policy(0.5),
    "weighted_random": WeightedLoadBalancer(pageview_file), 
    "adaptive_weighted_random": AdaptiveWeightedLoadBalancer(timestamp_weights_file), 
    "weighted_longest_queue": WeightedLongestQueueLoadBalancer(pageview_file),
    "longest_queue": LongestQueueLoadBalancer(),
    "random": RandomLoadBalancer(),
    "round_robin": RoundRobinLoadBalancerFix(),
    "weighted_round_robin": WeightedRoundRobin(pageview_file, keys)
}

def run_once(
    out_path: str,
    prioritization_policy: str,
    load_shedding_policy: str,
    keys: List[str],
    per_key_records_per_second: int,
    total_runtime_s: float,
    model_runtime_constant: float,
    key_selection_policy: str
):

    env = simpy.Environment()

    source_to_window_queue = simpy.Store(env)
    windows_to_mapper_queue = {
        key: PerKeyPriorityQueue(
            env,
            processing_policy=policies[prioritization_policy],
            load_shedding_policy=policies[load_shedding_policy],
        )
        for key in keys
    }

    JSONSource(
        env,
        records_per_sec_per_key=per_key_records_per_second,
        num_keys=len(keys),
        next_queue=source_to_window_queue,
        total_run_time=total_runtime_s,
        data_file=stream_edits_file,
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
        key_selection_policy_cls=policies[key_selection_policy],
        keys=keys,
    )
    env.run(until=total_runtime_s)

    plan = m.ready_time_to_batch
    with open(out_path, "w") as f:
        json.dump(plan, f)


if __name__ == "__main__":

    # argument flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--send_rate", type=int)
    parser.add_argument("--model_runtime", type=float)
    parser.add_argument("--total_runtime", type=float, default=len(edits))
    parser.add_argument("--event_policy", type=str)
    parser.add_argument("--key_policy", type=str)
    parser.add_argument("--load_shedding_policy", type=str)
    args = parser.parse_args()

    plan_name = f"{plan_dir}/plan-{args.key_policy}_{args.event_policy}-{args.load_shedding_policy}-{args.model_runtime}-{args.send_rate}"
    out_path = f"{plan_name}.json"
    print(out_path)
    run_once(
        out_path=out_path,
        prioritization_policy=args.event_policy,
        load_shedding_policy=args.load_shedding_policy,
        keys=keys,
        per_key_records_per_second=args.send_rate,
        total_runtime_s=args.total_runtime,
        model_runtime_constant=args.model_runtime,
        key_selection_policy=args.key_policy,
    )
    log_plans(run, config, plan_dir)


    # load sheding: random, drop short edits
    # prioritization: prioritize most recent version
    # cross-key prioritzation: historical page views,
    # policies
    #prioritization_policies = ["lifo"]  # ["fifo", "lifo"]
    ##key_selection_policies = ["adaptive_weighted_random", "weighted_round_robin", "weighted_random", "weighted_longest_queue", "longest_queue", "random", "round_robin"]
    #key_selection_policies = ["round_robin"]
    #load_shedding_policies = ["always_process"]
    ##model_runtimes = [0.01, 0.05, 0.1, 1, 5, 10]  # [0.000001, 0.00001, 0.0000001, 0.000000001, 0]
    #model_runtimes = [0.02, 0.05, 0.07]  # [0.000001, 0.00001, 0.0000001, 0.000000001, 0]
    #records_per_second = [100]

    #output_files = []

    #for key_selection in key_selection_policies:
    #    for prio_policy in prioritization_policies:
    #        for load_shed_policy in load_shedding_policies:
    #            for runtime in model_runtimes:
    #                for rate in records_per_second:

    #                    out_path = f"{plan_dir}/plan-{key_selection}_{prio_policy}-{load_shed_policy}-{runtime}-{rate}.json"
    #                    print("running", out_path, runtime)
    #                    run_once(
    #                        out_path,
    #                        prio_policy,
    #                        load_shed_policy,
    #                        keys,
    #                        per_key_records_per_second=rate,
    #                        total_runtime_s=len(edits),
    #                        model_runtime_constant=runtime,
    #                        key_selection_policy=key_selection,
    #                    )

    #                    output_files.append(out_path)
    #                    print("DONE", out_path)
    #for f in output_files:
    #    print(f)
    #open("plans.txt", "w").write("\n".join(output_files))
