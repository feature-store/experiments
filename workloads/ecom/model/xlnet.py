from functools import partial
import os

import torch
import transformers4rec.torch as t4r
from merlin_standard_lib import Schema, Tag
from transformers4rec.torch.trainer import HFWrapper
from transformers4rec.torch.utils.examples_utils import wipe_memory


def create_model(schema_file, ckpt_dir):
    schema = Schema().from_proto_text(schema_file)
    schema = schema.select_by_tag(Tag.ITEM_ID)
    item_id_col = schema.select_by_tag(Tag.ITEM_ID).column_names[0]
    col_names = schema.column_names

    embeddings_initializers = {}
    for col in col_names:
        if col == item_id_col:
            std = 0.09
        else:
            std = 0.015
        embeddings_initializers[col] = partial(torch.nn.init.normal_, mean=0.0, std=std)

    input_module = t4r.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        aggregation="concat",
        d_output=448,
        post=["layer-norm"],
        embedding_dims={item_id_col: 448},
        embedding_dim_default=448,
        infer_embedding_sizes=True,
        infer_embedding_sizes_multiplier=3.0,
        embeddings_initializers=embeddings_initializers,
        continuous_soft_embeddings=(False),
        soft_embedding_cardinality_default=(0),
        soft_embedding_dim_default=0,
        masking="mlm",
        mlm_probability=0.1,
    )
    prediction_task = t4r.NextItemPredictionTask(
        hf_format=True,
        weight_tying=True,
    )

    model_config = t4r.XLNetConfig.build(
        total_seq_length=20,
        d_model=448,
        n_head=8,
        n_layer=2,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dropout=0.0,
        pad_token=0,
        summary_type="last",
        attn_type="bi",
    )
    model = model_config.to_torch_model(input_module, prediction_task)

    model = HFWrapper(model)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu"))
    model = model.module
    model.eval()

    wipe_memory()
    return model


def create_backbone(schema_file, ckpt_dir):
    model = create_model(schema_file, ckpt_dir)
    body = model.heads[0].body
    return body


def create_prediction_head(schema_file, ckpt_dir):
    model = create_model(schema_file, ckpt_dir)
    pred_head = model.heads[0].prediction_task_dict["next-item"].pre
    return pred_head


def create_embedding_table(schema_file, ckpt_dir):
    pred_head = create_prediction_head(schema_file, ckpt_dir)
    embedding_table = pred_head.module.item_embedding_table
    return embedding_table
