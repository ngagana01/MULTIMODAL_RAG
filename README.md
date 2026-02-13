
# Multimodal Enterprise RAG

Advanced multimodal retrieval-augmented generation system supporting:

• PDFs + TXT  
• Images  
• Hybrid search  
• Streaming-ready architecture  
• Memory aware answers  

## Run

pip install -r requirements.txt
streamlit run app.py

## Features
- semantic + keyword retrieval
- FAISS vector DB
- hallucination guardrails
- latency tracking
- caching
- reranking
- session memory
- configurable pipeline

## Architecture
Upload → Chunk → Embed → Index → Retrieve → Rerank → LLM → Answer

## Live Demo
The application is deployed and accessible here:

https://multimodalrag-7amwvefymphrzhk8y7ybvf.streamlit.app/
