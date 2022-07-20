import json 
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
flags.DEFINE_integer(
    "updates",
    default=10,
    help="Updates per timestep"
)

class Queue: 

    def __init__(self): 
        self.queue = [] 

    def push(self, key, value): 
        self.queue.append(value)

    def pop(self): 
        if len(self.queue) == 0: 
            return None 

        event = self.queue[0]
        self.queue = self.queue[1:]
        return event

class Simulator: 

    def __init__(self, updates): 
        
        result_dir = "/data/wooders/ralf-vldb/results/wikipedia"
        dataset_dir = "/data/wooders/ralf-vldb/datasets/wikipedia"
        init_data_file = f"{dataset_dir}/init_data.json"
        stream_questions_file = f"{dataset_dir}/question_stream.json"
        stream_edit_file = f"{dataset_dir}/edit_stream.json"

        self.tmp_dir = "/data/wooders/ralf-vldb/results/wikipedia/tmp"
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
        self.embedding_version = json.load(open(init_data_file))
        for doc_id, diff in tqdm(self.embedding_version.items()):
            self.process_update(doc_id, diff["file"], init=True)

        # create update queue
        self.queue = Queue()

        self.updates = updates

 
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
        pred = self.stream_model.predict_single_doc(doc_questions_file, revid, doc_id)
        print("PREDICTION", pred)
        return pred


    def process_update(self, doc_id, filename, init=False): 

        # TODO: add cache

        revid_old = filename.replace(".json", "").split("_")[1]
        revid = filename.replace(".json", "").split("_")[0]

       # get updated embedding/passage data
        if init: 
            embedding_filename = self.embedding_path(revid, version="_orig")
        else:
            embedding_filename = self.embedding_path(revid, version="_new")

        data = json.load(open(os.path.join(self.rev_dir, filename)))
        timestamp = data["timestamp"]
        title = data["title"]
        sents = [d["sent_b"] for d in data["diffs"][0]]

        # load embedding
        embedding_data = pickle.load(open(self.embedding_path(revid, version="_new"), "rb"))
        passage_embeddings = embedding_data["embeddings"]
        passage_texts = embedding_data["passages"]

        # create input files 
        contex_file = f"{self.tmp_dir}/dpr_ctx_{revid}_{doc_id}"
        text_file = f"{self.tmp_dir}/passages_{revid}_{doc_id}.tsv"

        text_df = pd.DataFrame(
            [[i, passage_texts[i], "", doc_id] for i in range(len(passage_texts))]
        )
        text_df.to_csv(text_file, sep="\t", index=False, header=False)

        passage_ctx = []
        for i in range(len(passage_embeddings)):
            passage_ctx.append([i, passage_embeddings[i]])
        pickle.dump(np.array(passage_ctx, dtype=object), open(contex_file, "wb"))

        assert len(passage_ctx) == len(passage_texts)
        assert len(passage_embeddings) == len(passage_texts)

        print(f"update {doc_id} to {revid}")
        self.embedding_version[doc_id] = revid


    def run(self): 

        plan_results = [] #defaultdict(list)
        for ts in range(len(self.questions)): 
            timestep = ts / 100

            print(self.questions[ts])
            print(self.edits[ts])


            # process edits
            for doc_id, doc_questions in self.questions[ts].items(): 
                pred = self.predict_single(doc_id, doc_questions, ts)
                plan_results += pred

            # process questions:
            for key, revs in self.edits[ts].items(): 
                for rev in revs: 
                    self.queue.push(key, {"doc_id": key, "filename": rev, "ts": timestep})

            # process updates
            for _ in range(self.updates): 
                event = self.queue.pop()
                if not event: break
                self.process_update(event["doc_id"], event["filename"])

            if len(plan_results) > 0:

                top1, top5, top10 = 0, 0, 0
                for result in plan_results:
                    top1 += sum(result['doc_hits'][:1])
                    top5 += sum(result['doc_hits'][:5])
                    top10 += sum(result['doc_hits'][:10])
                top1 = top1 / len(plan_results)
                top5 = top5  / len(plan_results)
                top10 = top10 / len(plan_results)

                results = {'top1': top1,
                           'top5': top5,
                           'top10': top10}
                print("RESULT", results)

                







def main(argv):

    sim = Simulator(10) 
    sim.run()
    #runtime = [24, 12, 4, 2, 1, 0]
    #runtime = [4, 6, 8, 12, 24] #[1, 2, 3]
    policy = ["round_robin", "total_error", "batch"]
    #runtime = [0, 1000000]
    runtime = [0.05, 0.02] #0.5, 0.2, 0.1]
    name = f"yahoo_A1_window_{FLAGS.window_size}_keys_{FLAGS.num_keys}_length_{FLAGS.max_len}"

    result_dir = use_results(name)
    dataset_dir = use_dataset("yahoo/A1")

    # aggregate data structures
    results_df = pd.DataFrame()
    updates_df = pd.DataFrame()
    df_all = pd.DataFrame()

    data = read_data(dataset_dir)
    
    for r in runtime: 
        for p in policy: 

            try:
                update_times, df = simulate(data, start_ts=FLAGS.window_size, runtime=r, policy=p)
            except Exception as e:
                print(e) 
                continue
            e = error(df)
            s = df.staleness.mean()
            u = sum([len(v) for v in update_times.values()])
           
            r_df = pd.DataFrame([[r, p, e, s, u]])
            r_df.columns = ["runtime", "policy", "total_error", "average_staleness", "total_updates"]
            u_df = pd.DataFrame([
                [r, p, k, i, update_times[k][i]]
                for k, v in update_times.items() for i in range(len(v))
            ])
           
            # write experiment CSV
            folder = f"{p}_{r}_A1"
            os.makedirs(f"{result_dir}/{folder}", exist_ok=True)
            df.to_csv(f"{result_dir}/{folder}/simulation_predictions.csv")
            r_df.to_csv(f"{result_dir}/{folder}/simulation_result.csv")
            u_df.to_csv(f"{result_dir}/{folder}/simulation_update_time.csv")
            print(u_df)
            u_df.columns = ["runtime", "policy", "key", "i", "time"]
            u_df.to_csv(f"{result_dir}/{folder}/simulation_update_time.csv")
           
            # aggregate data 
            df_all = pd.concat([df_all, df])
            results_df = pd.concat([results_df, r_df])
            updates_df = pd.concat([updates_df, u_df])
            print("done", folder)

	
            
    results_df.to_csv(f"{result_dir}/simulation_results.csv")
    while True:
        try:
            log_results(name)
            break
        except Exception as e:
            print(e) 
            time.sleep(5)



if __name__ == "__main__":
    app.run(main)
