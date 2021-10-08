import configparser
from typing import List
import pickle
import shutil
from tqdm import tqdm
import time
import numpy as np
import json
import pandas as pd
import argparse
from statsmodels.tsa.seasonal import STL, DecomposeResult
import json
import os
from collections import defaultdict
from multiprocessing import Pool
import torch
from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    load_states_from_checkpoint,
    get_model_obj,
    move_to_device,
)
from dpr.utils.data_utils import Tensorizer


"""

Script for evaluating plans for the wikipedia edit stream dataset. Writes output files which need to be processed by DPR script. 

Download require data: 
    * rev_dir = s3://feature-store-datasets/wikipedia/edit_diffs/
    * init_data_file = s3://feature-store-datasets/wikipedia/simulation/init_data.json
    * questions_file = ??
    * model_file = s3://feature-store-datasets/wikipedia/models/bert-base-encoder.cp

Upload results: 
    * exp_dir = s3://feature-store-datasets/wikipedia/simulation_output/
"""
# simulation data

config = configparser.ConfigParser()
config.read("config.yml")
plan_dir = config["simulation"]["plan_dir"]
init_data_file = config["simulation"]["init_data_file"]
stream_edits_file = config["simulation"]["stream_edits_file"]
stream_questions_file = config["simulation"]["stream_questions_file"]

data_dir = config['files']['data_dir']
rev_dir = config['directory']['diff_dir'] 
embedding_dir = config['directory']['embedding_dir'] 
exp_dir = config['directory']['exp_dir'] 
model_file = config['files']['model_file'] 

# Create parser
parser = argparse.ArgumentParser(description="Specify experiment config")
parser.add_argument("--offline-plan-path", type=str)
parser.add_argument("--embed", default=False, action="store_true")
args = parser.parse_args()


class Retriever:
    def __init__(self):

        # parser = argparse.ArgumentParser(description="")
        add_encoder_params(parser)
        add_tokenizer_params(parser)
        add_cuda_params(parser)
        args = parser.parse_args()

        setup_args_gpu(args)

        saved_state = load_states_from_checkpoint(model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

        self.tensorizer, self.encoder, _ = init_biencoder_components(
            args.encoder_model_type, args, inference_only=True
        )

        self.encoder = self.encoder.ctx_model

        self.encoder, _ = setup_for_distributed_mode(
            self.encoder,
            None,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level,
        )
        self.encoder.eval()

        model_to_load = get_model_obj(self.encoder)

        prefix_len = len("ctx_model.")
        ctx_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith("ctx_model.")
        }
        model_to_load.load_state_dict(ctx_state)
        self.device = args.device

    def predict(self, text):

        st = time.time()
        batch_token_tensors = [self.tensorizer.text_to_tensor(text)]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), self.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), self.device)
        ctx_attn_mask = move_to_device(
            self.tensorizer.get_attn_mask(ctx_ids_batch), self.device
        )
        with torch.no_grad():
            _, embedding, _ = self.encoder(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        embedding = embedding.cpu().numpy()
        return embedding


def assign_timestamps_min(ts):
    # take in unix timestamp - covert to integer
    start_ts = 1628131044000000000  # don't change
    delta = ts - start_ts
    if delta < 0:
        return None

    return int(delta / (60 * 1000000000))


def embed_passages(sents, retriever_model, num_sent_in_pass=10):
    passages = []
    embeddings = []
    for i in range(0, len(sents), num_sent_in_pass):
        passages.append(" ".join(sents[i : i + num_sent_in_pass]))
        embeddings.append(retriever_model.predict(passages[-1]))
    return passages, embeddings


def sents_to_passages(sents, num_sent_in_pass=10):
    passages = []

    for i in range(0, len(sents), num_sent_in_pass):
        passages.append(" ".join(sents[i : i + num_sent_in_pass]))
    return passages


def embedding_path(revid, version="_new"):
    return os.path.join(embedding_dir, f"{revid}{version}.pkl")


def offline_eval(plan_json_path, exp_id, compute_embeddings=True):

    # only process subset of keys
    keys = ["51150040"]
    filter_keys = False

    # retriever_model = Retriever()
    # print("Created retriever")

    # compute initial passage embeddings for each document
    init_data = json.load(open(init_data_file))
    init_state = {}
    staleness = []
    for key in tqdm(init_data.keys()):

        if filter_keys and key not in keys:
            continue
        sents = init_data[key]["sents"]
        revid = init_data[key]["revid"]

        print(init_data_file)
        print(init_data[key]["file"])
        embedding_data = pickle.load(open(embedding_path(revid, version="_orig"), "rb"))
        embeddings = embedding_data["embeddings"]
        passages = sents_to_passages(sents)
        if not len(passages) == len(embeddings):
            print(f"passage {len(passages)} embeddings {len(embeddings)}")
            print(len(embedding_data["passages"]))
            print(revid)
            print("diff file", init_data[key]["file"])
            print("embedding file", embedding_data["file"])
            print(embedding_data["timestamp"])
            print(len(sents))
            return

        init_state[key] = {
            "passages": passages,
            "embeddings": embeddings,
            "rev": "init",
        }

    print(f"Created init state for {len(init_state.keys())} keys")

    # compute passage embeddings for each timestep (using plan)
    embed_versions = {"0": init_state}
    plan = json.load(open(plan_json_path))
    embed_version_keys: List[str] = list(plan.keys())
    count = 0
    missing = set([])
    print("looping keys", len(embed_version_keys))
    for version in tqdm(embed_version_keys):
        state = {}
        for task in plan[version]:
            #print("task", task, version)
            rev_file = task[0]
            doc_id = task[1]
            # doc_id = task[2]
            # rev_file = task[3]
            # if filter_keys and doc_id not in keys:
            #    continue
            data = json.load(open(os.path.join(rev_dir, rev_file)))
            timestamp = data["timestamp"]
            title = data["title"]
            sents = [d["sent_b"] for d in data["diffs"][0]]
            revid = rev_file.replace(".json", "").split("_")[0]
            embedding_filename = embedding_path(revid, version="_new")
            assert os.path.exists(
                embedding_filename
            ), f"Missing revid {embedding_filename}"
            if os.path.exists(embedding_filename):
                embedding_data = pickle.load(
                    open(embedding_path(revid, version="_new"), "rb")
                )
                # assert embedding_data["timestamp"] == timestamp
                embeddings = embedding_data["embeddings"]
                passages = embedding_data["passages"]
                assert len(passages) == len(
                    sents_to_passages(sents)
                ), f"Inconsistent passage len {len(passages)}, {len(sents_to_passages(sents))}"
                # passages = sents_to_passages(sents)
                assert len(passages) == len(embeddings)
            else:
                missing.add(doc_id)
                continue
            # print("fitting", timestamp, version, doc_id, rev_file)
            count += 1
            state[doc_id] = {
                "passages": passages,
                "embeddings": embeddings,
                "rev": rev_file,
            }

        # save version
        embed_versions[version] = state
    print("EMBED", embed_versions.keys())
    print("Num refits", count, len(missing))

    # returns latest version of document embeddings for timestep/key
    def get_latest_embedding(timestep, doc_id):

        latest = 0
        for version in embed_versions.keys():
            version = float(version)
            if (
                float(timestep) >= version
                and version > latest
                and doc_id in embed_versions[str(version)]
            ):
                latest = version
        #print(doc_id, "latest", timestep, latest, timestep - latest)
        assert (
            doc_id in embed_versions[str(latest)]
        ), f"Missing doc id {doc_id} {latest} {doc_id in init_data}"
        doc_version = embed_versions[str(latest)][doc_id]
        assert latest <= timestep
        return (
            doc_version["passages"],
            doc_version["embeddings"],
            doc_version["rev"],
            latest,
        )

    # create experiment directory
    directory = os.path.join(exp_dir, exp_id)
    if os.path.isdir(directory):
        print("Removing", directory)
        shutil.rmtree(directory)
    print("Creating", directory)
    os.mkdir(directory)

    # get simulation data questions
    questions = json.load(open(stream_questions_file))
    print("processing questions", len(questions))
    print("directory", directory)

    # Get embedding version for each query, write outputs
    for ts in tqdm(range(len(questions))):
        timestep = ts / 100  # TODO: Watch out!! can change and mess up experiment
        # print(ts, timestep)

        for doc_id in questions[ts].keys():

            # not considered in edits
            if doc_id not in init_data:
                print("missing", doc_id)
                # print(init_data.keys())
                continue

            # get current embedding and write
            passage_texts, passage_embeddings, version, latest = get_latest_embedding(
                timestep, doc_id
            )
 
            # loop through questions
            doc_questions = questions[ts][doc_id]
            queries = []
            for q in doc_questions:
                question = q["question"]
                answer = q["answer"]
                assert (
                    str(q["doc_id"]) == doc_id
                ), f"doc id mismatch {q['doc_id']}, {doc_id}"
                assert (
                    q["ts_min"] == ts
                ), f"time mismatch {q['ts_min']}, {timestep}, {ts}"
                queries.append([question, [answer], doc_id])

                # append per query
                staleness.append(timestep - latest)

            # dump CTX/question script
            contex_file = f"{directory}/dpr_ctx_after_{int(ts)}_{doc_id}"
            text_file = f"{directory}/passages_{int(ts)}_{doc_id}.tsv"
            doc_questions_file = (
                f"{directory}/qa_{int(ts)}_{doc_id}.tsv"  # question, answer(s), doc id
            )
            doc_questions_df = pd.DataFrame(queries)
            doc_questions_df.to_csv(
                doc_questions_file, sep="\t", index=False, header=False
            )
            # write passage file - id, text, title, doc_id
            text_df = pd.DataFrame(
                [[i, passage_texts[i], "", doc_id] for i in range(len(passage_texts))]
            )
            text_df.to_csv(text_file, sep="\t", index=False, header=False)
            # write ctx file
            passage_ctx = []
            for i in range(len(passage_embeddings)):
                passage_ctx.append([i, passage_embeddings[i]])
            pickle.dump(np.array(passage_ctx, dtype=object), open(contex_file, "wb"))

            assert len(passage_ctx) == len(passage_texts)
            assert len(passage_embeddings) == len(passage_texts)

    print("done processing queries!", len(questions))
    print("staleness", np.array(staleness).mean())
    return directory


def main():

    plan_file = (
        args.offline_plan_path
    )  # "wiki-plans/plan-fifo-always_process-1-0.001-60.json"
    exp_id = os.path.basename(plan_file).replace(".json", "")
    
    output_dir = offline_eval(plan_file, exp_id, compute_embeddings=args.embed)
    log_wandb = False
    if log_wandb:
        import wandb

        run = wandb.init(job_type="create_simulation_output")
        artifact = wandb.Artifact(exp_id, type="dataset")
        artifact.add_folder(output_dir)


if __name__ == "__main__":
    main()
