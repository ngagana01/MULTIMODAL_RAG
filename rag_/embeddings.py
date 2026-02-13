import requests
import hashlib
from config import JINA_API_KEY, CACHE_PATH
from .utils import load_pickle, save_pickle

cache = load_pickle(CACHE_PATH) or {}

def _hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def embed(texts):
    new = []
    mapping = []

    for t in texts:
        h = _hash(t)
        if h not in cache:
            new.append(t)
            mapping.append(h)

    if new:
        res = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={"model":"jina-embeddings-v4","input":new}
        ).json()

        for h,vec in zip(mapping, res["data"]):
            cache[h] = vec["embedding"]

        save_pickle(cache, CACHE_PATH)

    return [cache[_hash(t)] for t in texts]
