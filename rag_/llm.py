import requests
from config import GROQ_API_KEY, LLM_MODEL

def generate(query, context):
    prompt = f"""
Answer ONLY using context.
If answer not found say "Not in document".

Context:
{context}

Question:
{query}
"""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model":LLM_MODEL,
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.2
        }
    )

    return r.json()["choices"][0]["message"]["content"]
