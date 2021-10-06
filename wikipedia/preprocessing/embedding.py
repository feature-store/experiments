from typing import List
import pickle
from tqdm import tqdm
import time
import numpy as np
import json
import pandas as pd
import argparse
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


class Retriever:
    def __init__(self, model_file):

        parser = argparse.ArgumentParser(description="")
        add_encoder_params(parser)
        add_tokenizer_params(parser)
        add_cuda_params(parser)
        args = parser.parse_args()

        setup_args_gpu(args)

        print(args)

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


def embed_passages(sents, retriever_model, num_sent_in_pass=10):
    passages = []
    embeddings = []
    for i in range(0, len(sents), num_sent_in_pass):
        passages.append(" ".join(sents[i : i + num_sent_in_pass]))
        embeddings.append(retriever_model.predict(passages[-1]))
    return passages, embeddings


def get_passages(sents, num_sent_in_pass=10):
    passages = []
    for i in range(0, len(sents), num_sent_in_pass):
        passages.append(" ".join(sents[i : i + num_sent_in_pass]))
    return passages


def generate_embeddings(model_file, diff_dir, embedding_dir):

    # create retriever
    retriever_model = Retriever(model_file)

    # loop through files
    index = 0
    # gpu = 2
    for filename in tqdm(os.listdir(diff_dir)):

        # index += 1
        # if index % 5 != gpu:
        #    continue

        new_id = filename.replace(".json", "").split("_")[0]
        old_id = filename.replace(".json", "").split("_")[1]

        for revid in [new_id, old_id]:

            data = json.load(open(os.path.join(diff_dir, filename)))

            if len(data["diffs"]) == 0:
                continue

            if revid == data["orig_id"]:
                sents = [d["sent_a"] for d in data["diffs"][0]]
                filepath = os.path.join(embedding_dir, revid + "_orig.pkl")
            elif revid == data["new_id"]:
                sents = [d["sent_b"] for d in data["diffs"][0]]
                filepath = os.path.join(embedding_dir, revid + "_new.pkl")

            if not os.path.exists(filepath):
                passages, embeddings = embed_passages(sents, retriever_model)

                pickle.dump(
                    {
                        "timestamp": data["timestamp"],
                        "passages": passages,
                        "file": filepath,
                        "embeddings": embeddings,
                    },
                    open(filepath, "wb"),
                )
