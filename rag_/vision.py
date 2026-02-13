import base64
import requests
from config import GROQ_API_KEY, VISION_MODEL

def describe_image(file):
    b64 = base64.b64encode(file.read()).decode()

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model":VISION_MODEL,
            "messages":[{
                "role":"user",
                "content":[
                    {"type":"text","text":"Describe this image in detail"},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
                ]
            }]
        }
    )

    return res.json()["choices"][0]["message"]["content"]
