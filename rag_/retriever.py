import numpy as np

class Retriever:
    def __init__(self, dim):
        self.vectors=[]
        self.texts=[]

    def add(self, vectors, texts):
        self.vectors.extend(vectors)
        self.texts.extend(texts)

    def search(self, vector, k=5):
        sims=[np.dot(vector,v) for v in self.vectors]
        idx=np.argsort(sims)[::-1][:k]
        return [self.texts[i] for i in idx]
