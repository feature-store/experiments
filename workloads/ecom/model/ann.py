import torch
import faiss


class ANNRetriever:

    def __init__(
        self,
        embedding,
        num_bits,
    ):
        self.embedding_dim = embedding.embedding_dim
        self.index = faiss.IndexLSH(self.embedding_dim, num_bits)

        embedding_vectors = embedding.weight.detach().numpy()
        self.index.add(embedding_vectors)

    def query(self, vectors, k=10):
        _, indices = self.index.search(vectors, k) 
        return indices
