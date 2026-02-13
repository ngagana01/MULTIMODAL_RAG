import time
import pickle
import os

def timer():
    return time.perf_counter()

def elapsed(start):
    return round(time.perf_counter() - start, 2)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
