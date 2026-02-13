import os

# API KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# MODELS
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL = "llama-3.3-70b-versatile"

# RAG SETTINGS
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
TOP_K = 5
RERANK_K = 3

# PATHS
INDEX_PATH = "faiss_index"
CACHE_PATH = "embed_cache.pkl"

# UI
APP_TITLE = "Multimodal Enterprise RAG"
