import faiss
import numpy as np
import os
from config import INDEX_PATH

class Retriever:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vectors, texts):
        arr = np.array(vectors).astype("float32")
        self.index.add(arr)
        self.texts.extend(texts)

    def search(self, vector, k=5):
        D,I = self.index.search(np.array([vector]).astype("float32"), k)
        return [self.texts[i] for i in I[0]]

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)

    def load(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
