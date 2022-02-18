import torch
import torch.nn as nn


class Predictor(nn.Module):

    def __init__(self, embedding_table):
        super().__init__()
        self.embedding_table = embedding_table

    def forward(self, x, candidate_indicies=None):
        if candidate_indicies is not None:
            candidate_embeddings = self.embedding_table(candidate_indicies)
        else:
            candidate_embeddings = self.embedding_table.weight.unsqueeze(dim=0)
        logits = torch.matmul(x, candidate_embeddings.permute(0, 2, 1))
        return logits
