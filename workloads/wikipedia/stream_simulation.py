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
        self.past_updates = defaultdict(lambda: 1)
        self.pending = defaultdict(lambda: 0)

        # new baselines 
        if policy == "query_cold":
            self.past_queries = defaultdict(lambda: 100000000) 
        else:
            self.past_queries = defaultdict(lambda: 0) 


    def push(self, key, value): 
        self.queue[key] = value
        self.last_key[key] = time.time()
        self.pending[key] += 1
        
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
            key = self.arg_max(self.pending)
        elif self.policy == "min_past": 
            key = self.arg_max({key: 1/(self.past_updates.setdefault(key, 0)+1) for key in self.queue.keys()})
        elif self.policy == "round_robin": 
            key = self.arg_max(self.staleness)
        elif self.policy == "query_proportional" or self.policy == "query_cold": 
            key = self.arg_max(self.past_queries)
        elif self.policy == "batch":
            key = self.arg_max(self.staleness) # doensn't matter
        elif self.policy == "random": 
            options = [k for k in self.queue.keys() if self.queue[k] is not None]
            if len(options) == 0: 
                key = (None, None)
            else:
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
        self.past_updates[key] += 1
        self.pending[key] = 0
        
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
            #self.total_error[key] += 1




class Simulator: 

    def __init__(self, policy, updates): 
        
        result_dir = "/data/wooders/ralf-vldb/results/wikipedia"
        dataset_dir = "/data/wooders/ralf-vldb/datasets/wikipedia"
        init_data_file = f"{dataset_dir}/init_data.json"
        stream_questions_file = f"{dataset_dir}/question_stream.json"
        stream_edit_file = f"{dataset_dir}/edit_stream.json"

        self.tmp_dir = "/data/wooders/ralf-vldb/results/wikipedia/tmp"
        self.embedding_dir = f"{result_dir}/swap_embeddings"
        self.rev_dir = f"{result_dir}/diffs_new"

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

        # filter out invalid revisions
        self.valid_revisions = set() 
        for q in self.questions: 
            for ql in q.values(): 
                for qll in ql: 
                    self.valid_revisions.add(str(qll["revid"]))

        print("Keys", len(self.embedding_version.keys()))

        # create update queue
        self.queue = Queue(policy)
        self.policy = policy 

        self.updates = updates
        self.policy = policy 

        self.cache = {}


 
    def embedding_path(self, revid, version="_new"):
        return os.path.join(self.embedding_dir, f"{revid}{version}.pkl")

    def cache_key(self, question, answer, doc_id, revid): 
        return str(question) + str(answer) + str(doc_id) + str(revid)

    def predict_single(self, doc_id, questions, ts): 

        revid = self.embedding_version[doc_id]

        queries = []
        input_questions = []
        for q in questions: 
            question = q["question"]
            answer = q["answer"]

            if self.cache_key(question, answer, doc_id, revid) not in self.cache: 
                queries.append([question, [answer], doc_id])
                input_questions.append(q)
       
        if len(input_questions) > 0:
            doc_questions_file = f"{self.tmp_dir}/qa_{self.policy}_{self.updates}_{int(ts)}_{doc_id}"
            doc_questions_df = pd.DataFrame(queries)
            doc_questions_df.to_csv(
                doc_questions_file, sep="\t", index=False, header=False
            )

            print("Using version", revid, "init", self.init_data[doc_id]["revid"])
            pred = self.stream_model.predict_single_doc(doc_questions_file, revid, doc_id)

            for q, p in zip(input_questions, pred): 
                p["doc_id"] = q["doc_id"] 
                p["revid"] = q["revid"]
                p["actual_revid"] = revid
                p["ts"] = ts

                self.cache[self.cache_key(q["question"], q["answer"], q["doc_id"], revid)] = p

        # collect and return predictions
        predictions = []
        for q in questions: 
            predictions.append(self.cache[self.cache_key(q["question"], q["answer"], q["doc_id"], revid)])
        return predictions


    def process_update(self, doc_id, filename, init=False): 

        # TODO: add cache

        revid_old = filename.replace(".json", "").split("_")[1]
        revid = filename.replace(".json", "").split("_")[0]

        print("processs update", filename)

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

        if True: #not os.path.exists(contex_file):

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
        #for ts in range(len(self.questions)): 
        #steps = len(self.questions) 
        steps = 10000
        for ts in range(steps):

            # update budget
            budget += self.updates

            # process events
            print(ts, "Updates", len(list(self.edits[ts].items())), "budget", budget, "queue", self.queue.size())
            for key, revs in self.edits[ts].items(): 
                if key not in self.embedding_version: continue # skip 
                for rev in revs: 
                    revid = rev.replace(".json", "").split("_")[0]
                    #if revid not in self.valid_revisions: 
                    #    print("skipping", revid) 
                    #    continue
                    self.queue.push(key, {"doc_id": key, "filename": rev, "ts": ts})

            # process updates
            if (self.policy == "batch" and budget >= self.queue.size()) or self.policy != "batch": 
                while budget >= 1: 
                    event = self.queue.pop(ts)
                    if event: 
                        print(f'{ts}: update to {event["filename"]}')
                        self.process_update(event["doc_id"], event["filename"])
                        event["update_time"] = ts
                        updates.append(event)
                        budget -= 1
                    else: 
                        break
                if budget >= 1:
                    budget = 0

            # process questions
            for doc_id, doc_questions in self.questions[ts].items(): 
                if doc_id not in self.embedding_version: continue # skip

                # filter out questions for future revisions
                #doc_questions = [q for q in doc_questions if str(q["revid"]) == self.curr_rev[doc_id]]
                if len(doc_questions) == 0: continue

               
                pred = self.predict_single(doc_id, doc_questions, ts)
                for p in pred: 

                    self.queue.error(doc_id, 1-max(p['doc_hits'][:1]))
                    # append results
                plan_results += pred

            # increment queue timestep
            self.queue.timestep()


            if (len(plan_results) > 0 and len(plan_results) % 100 == 0) or ts == steps - 1:

                pd.DataFrame(plan_results).to_csv(f"{results_dir}/step_{steps}_results_{self.policy}_{self.updates}.csv")
                pd.DataFrame(updates).to_csv(f"{results_dir}/step_{steps}_updates_{self.policy}_{self.updates}.csv")

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
    result_dir = "/data/wooders/ralf-vldb/results/wikipedia/stream_simulation/"
    sim.run(result_dir)



if __name__ == "__main__":
    app.run(main)
