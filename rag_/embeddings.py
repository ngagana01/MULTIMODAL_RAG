import requests
import hashlib
from config import JINA_API_KEY, CACHE_PATH
from .utils import load_pickle, save_pickle

cache = load_pickle(CACHE_PATH) or {}

def _hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def embed(texts):
    if not texts:
        return []

    new = []
    mapping = []

    for t in texts:
        h = _hash(t)
        if h not in cache:
            new.append(t)
            mapping.append(h)

   
    if new:

        if not JINA_API_KEY:
            raise ValueError("JINA_API_KEY is missing in config.py or environment variables.")

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "jina-embeddings-v4",
                "input": new
            }
        )

       
        if response.status_code != 200:
            raise Exception(f"Embedding API failed ({response.status_code}): {response.text}")

        data = response.json()

        
        if "data" not in data:
            raise Exception(f"Unexpected API response: {data}")

        
        for h, vec in zip(mapping, data["data"]):
            cache[h] = vec["embedding"]

        save_pickle(cache, CACHE_PATH)

    
    return [cache[_hash(t)] for t in texts]
