import streamlit as st
from config import *
from rag_.chunking import chunk_text
from rag_.embeddings import embed
from rag_.retriever import Retriever
from rag_.reranker import rerank
from rag_.vision import describe_image
from rag_.llm import generate
from rag_.memory import ChatMemory
from rag_.utils import timer, elapsed
from pypdf import PdfReader

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

memory = ChatMemory()

# SIDEBAR
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K",1,10,TOP_K)
mode = st.sidebar.selectbox("Retrieval Mode",["text","image","both"])

# FILE UPLOAD
doc = st.file_uploader("Upload TXT or PDF")
img = st.file_uploader("Upload Image")

if doc:
    text = ""
    if doc.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(doc)
        for p in reader.pages:
            text += p.extract_text()
    else:
        text = doc.read().decode()

    chunks = chunk_text(text,CHUNK_SIZE,CHUNK_OVERLAP)
    vecs = embed(chunks)

    retriever = Retriever(len(vecs[0]))
    retriever.add(vecs,chunks)

    st.success(f"{len(chunks)} chunks indexed")

# IMAGE
img_desc = ""
if img and mode!="text":
    with st.spinner("Analyzing image..."):
        img_desc = describe_image(img)
        st.info(img_desc)

# QUERY
q = st.text_input("Ask question")

if q:
    start = timer()

    ctx = ""
    if doc and mode!="image":
        q_vec = embed([q])[0]
        docs = retriever.search(q_vec, top_k)
        docs = rerank(q, docs)
        ctx += "\n".join(docs)

    if img_desc and mode!="text":
        ctx += "\n"+img_desc

    ctx += "\n"+memory.context()

    ans = generate(q, ctx)
    memory.add(q,ans)

    st.markdown("### Answer")
    st.write(ans)

    st.caption(f"Latency: {elapsed(start)}s")
