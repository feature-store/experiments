import json 
import random
import pickle
import time
from collections import defaultdict
import os
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from absl import app, flags

from tqdm import tqdm

#from workloads.util import use_results, use_dataset, read_config, log_dataset, log_results

import sys
sys.path.insert(0, "../DPR/")

from dense_retriever_stream import StreamingModel

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_float(
    "updates",
    default=0,
    help="Updates per timestep"
)
flags.DEFINE_string(
    "policy",
    default=None,
    help="Update policy"
)

class Queue: 
    
    """
    Event queue that selects group of user updates
    (note: we can have another implementation which triggers a bunch of updates together)
    """
    
    def __init__(self, policy):
        self.policy = policy 
        
        # metric tracking
        if policy == "total_error_cold": 
            self.total_error = defaultdict(lambda: 1000000)
        else: 
            self.total_error = defaultdict(lambda: 0)
        self.queue = defaultdict(lambda: None)
        self.staleness = defaultdict(lambda: 0)
        self.last_key = defaultdict(lambda: 0)
        self.past_updates = defaultdict(lambda: 0)

        # new baselines 
        self.past_queries = defaultdict(lambda: 0) 

    def push(self, key, value): 
        self.queue[key] = value
        self.last_key[key] = time.time()
        
    def arg_max(self, data_dict): 
        max_key = None
        max_val = None
        valid_keys = 0
        for key in self.queue.keys():
            
            # cannot select empty queue 
            if self.queue[key] is None:
                continue 
                
            valid_keys += 1
            value = data_dict[key]
            if max_key is None or max_val <= value: 
                assert key is not None, f"Invalid key {data_dict}"
                max_key = key
                max_val = value
        return max_key, max_val
        
        
    def choose_key(self): 
        if self.policy == "total_error_cold" or self.policy == "total_error":
            key = self.arg_max(self.total_error)
        elif self.policy == "last_query":
            key = self.arg_max(self.last_key)
        elif self.policy == "max_pending":
            key = self.arg_max({key: len(self.queue[key]) for key in self.queue.keys()})
        elif self.policy == "min_past": 
            key = self.arg_max({key: 1/(self.past_updates.setdefault(key, 0)+1) for key in self.queue.keys()})
        elif self.policy == "round_robin": 
            key = self.arg_max(self.staleness)
        elif self.policy == "query_proportional": 
            key = self.arg_max(self.past_queries)
        elif self.policy == "batch":
            key = self.arg_max(self.staleness) # doensn't matter
        elif self.policy == "random": 
            options = [k for k in self.queue.keys() if self.queue[k] is not None]
            key = (random.choice(options), 0)
        else: 
            raise ValueError("Invalid policy")
       
        assert key is not None or sum([len(v) for v in self.queue.values()]) == 0, f"Key is none, {self.queue}"
        return key 
    
    def pop(self, ts): 
        key, score = self.choose_key()
        if key is None:
            return None 
        event = self.queue[key]
        self.queue[key] = None

        self.staleness[key] = 0
        self.total_error[key] = 0
        self.past_queries[key] = 0
        
        return event

    def size(self): 
        size = 0
        for key in self.queue.keys(): 
            if self.queue[key]:
                size += 1
        return size

    def error(self, key, error): 
        self.past_queries[key] += 1
        self.total_error[key] += error

    def timestep(self): 
        for key in self.queue.keys(): 
            self.staleness[key] += 1


class Simulator: 

    def __init__(self, policy, updates): 
        
        result_dir = "/data/jeffcheng1234/ralf-vldb/results/wikipedia"
        dataset_dir = "/data/jeffcheng1234/ralf-vldb/datasets/wikipedia"
        init_data_file = f"{dataset_dir}/init_data.json"
        stream_questions_file = f"{dataset_dir}/question_stream.json"
        stream_edit_file = f"{dataset_dir}/edit_stream.json"

        self.tmp_dir = "/data/jeffcheng1234/ralf-vldb/results/wikipedia/tmp"
        self.embedding_dir = f"{result_dir}/embeddings"
        self.rev_dir = f"{result_dir}/diffs"

        # create model file
        model_file = f"{result_dir}/models/bert-base-encoder.cp"
        self.stream_model = StreamingModel(model_file)

        # load question stream
        self.questions = json.load(open(stream_questions_file))

        # load edit stream
        self.edits = json.load(open(stream_edit_file))

        # load initial data
        self.init_data = json.load(open(init_data_file))
        self.embedding_version = {}
        for doc_id, diff in tqdm(self.init_data.items()):
            #print("REV", diff["revid"])
            self.process_update(doc_id, diff["file"], init=True)
            self.embedding_version[doc_id] = diff["revid"]

        # create update queue
        self.queue = Queue(policy)
        self.policy = policy 

        self.updates = updates
        self.policy = policy 

 
    def embedding_path(self, revid, version="_new"):
        return os.path.join(self.embedding_dir, f"{revid}{version}.pkl")

    def predict_single(self, doc_id, questions, ts): 

        queries = []
        for q in questions: 
            question = q["question"]
            answer = q["answer"]
            queries.append([question, [answer], doc_id])
        
        doc_questions_file = f"{self.tmp_dir}/qa_{int(ts)}_{doc_id}.tsv"
        doc_questions_df = pd.DataFrame(queries)
        doc_questions_df.to_csv(
            doc_questions_file, sep="\t", index=False, header=False
        )

        revid = self.embedding_version[doc_id]
        #print("Using version", revid, "init", self.init_data[doc_id]["revid"])
        pred = self.stream_model.predict_single_doc(doc_questions_file, revid, doc_id)
        for q, p in zip(questions, pred): 
            p["doc_id"] = q["doc_id"]
            p["revid"] = q["revid"]
            p["actual_revid"] = revid
            p["ts"] = ts
        return pred


    def process_update(self, doc_id, filename, init=False): 

        # TODO: add cache

        revid_old = filename.replace(".json", "").split("_")[1]
        revid = filename.replace(".json", "").split("_")[0]

        # get updated embedding/passage data
        data = json.load(open(os.path.join(self.rev_dir, filename)))
        timestamp = data["timestamp"]
        title = data["title"]

        if init: # use old version
            revid = revid_old
            #embedding_filename = self.embedding_path(revid, version="_orig")
            embedding_filename = self.embedding_path(revid, version="_orig")
            sents = [d["sent_a"] for d in data["diffs"][0]]
        else:
            embedding_filename = self.embedding_path(revid, version="_new")
            sents = [d["sent_b"] for d in data["diffs"][0]]
            print(f"update {doc_id} to {revid}")
            self.embedding_version[doc_id] = revid

        # load embedding
        #print(embedding_filename)
        embedding_data = pickle.load(open(embedding_filename, "rb"))
        passage_embeddings = embedding_data["embeddings"]
        passage_texts = embedding_data["passages"]
        assert len(passage_embeddings) == len(passage_texts)

        assert len(passage_embeddings) > 0, f"Empty embeddings {filename}, {embedding_filename}"

        # create input files 
        contex_file = f"{self.tmp_dir}/dpr_ctx_{revid}_{doc_id}"
        text_file = f"{self.tmp_dir}/passages_{revid}_{doc_id}.tsv"

        if not os.path.exists(contex_file):

            text_df = pd.DataFrame(
                [[i, passage_texts[i], "", doc_id] for i in range(len(passage_texts))]
            )
            text_df.to_csv(text_file, sep="\t", index=False, header=False)

            passage_ctx = []
            for i in range(len(passage_embeddings)):
                passage_ctx.append([i, passage_embeddings[i]])
            pickle.dump(np.array(passage_ctx, dtype=object), open(contex_file, "wb"))

            assert len(passage_ctx) == len(passage_texts)


        

    def run(self, results_dir): 


        plan_results = [] #defaultdict(list)
        updates = []
        budget = 0
        for ts in range(len(self.questions)): 

            # update budget
            budget += self.updates

            # process questions
            for doc_id, doc_questions in self.questions[ts].items(): 
                pred = self.predict_single(doc_id, doc_questions, ts)
                for p in pred: 

                    self.queue.error(doc_id, 1-max(p['doc_hits'][:1]))
                    # append results
                plan_results += pred

            # process questions:
            print("Updates", len(list(self.edits[ts].items())), "budget", budget)
            for key, revs in self.edits[ts].items(): 
                for rev in revs: 
                    self.queue.push(key, {"doc_id": key, "filename": rev, "ts": ts})

            # process updates
            if (self.policy == "batch" and budget >= self.queue.size()) or self.policy != "batch": 
                while budget > 1: 
                    event = self.queue.pop(ts)
                    if not event: break
                    self.process_update(event["doc_id"], event["filename"])
                    event["update_time"] = ts
                    updates.append(event)
                    budget -= 1

            # increment queue timestep
            self.queue.timestep()


            if len(plan_results) > 0 and len(plan_results) % 100 == 0:

                pd.DataFrame(plan_results).to_csv(f"{results_dir}/results_{self.policy}_{self.updates}.csv")
                pd.DataFrame(updates).to_csv(f"{results_dir}/updates_{self.policy}_{self.updates}.csv")

                top1, top5, top10 = 0, 0, 0
                doc_size = 0 
                for result in plan_results:
                    top1 += max(result['doc_hits'][:1])
                    top5 += max(result['doc_hits'][:5])
                    top10 += max(result['doc_hits'][:10])
                    doc_size += len(result['doc_hits'])
                top1 = top1 / len(plan_results)
                top5 = top5  / len(plan_results)
                top10 = top10 / len(plan_results)
                doc_size = doc_size / len(plan_results)

                results = {'top1': top1,
                           'top5': top5,
                           'top10': top10, 
                           'doc_size': doc_size }
                print(ts, "RESULT", results)


def main(argv):

    sim = Simulator(FLAGS.policy, FLAGS.updates) 
    result_dir = "/data/jeffcheng1234/ralf-vldb/results/wikipedia/stream_simulation/"
    sim.run(result_dir)



if __name__ == "__main__":
    app.run(main)
