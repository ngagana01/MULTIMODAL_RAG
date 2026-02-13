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

from PyPDF2 import PdfReader




st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)


if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chunks_loaded" not in st.session_state:
    st.session_state.chunks_loaded = False


st.sidebar.header("Settings")

top_k = st.sidebar.slider("Top Results", 1, 10, TOP_K)
mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["text", "image", "both"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Multimodal RAG System")



doc = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])



if doc and not st.session_state.chunks_loaded:
    text = ""

    try:
        if doc.name.endswith(".pdf"):
            reader = PdfReader(doc)
            for page in reader.pages:
                text += page.extract_text() or ""
        else:
            text = doc.read().decode("utf-8")

        if not text.strip():
            st.error("No readable text found in file.")
            st.stop()

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        with st.spinner("Generating embeddings..."):
            vectors = embed(chunks)

        retriever = Retriever(len(vectors[0]))
        retriever.add(vectors, chunks)

        st.session_state.retriever = retriever
        st.session_state.chunks_loaded = True

        st.success(f"Indexed {len(chunks)} chunks")

    except Exception as e:
        st.error(f"Processing error: {e}")
        st.stop()



img_description = ""

if img and mode != "text":
    try:
        with st.spinner("Analyzing image..."):
            img_description = describe_image(img)
        st.info(img_description)
    except Exception as e:
        st.error(f"Image analysis failed: {e}")



query = st.text_input("Ask a question about your data")

if query:
    start = timer()
    context = ""

    try:
        if st.session_state.retriever and mode != "image":
            q_vec = embed([query])[0]
            docs = st.session_state.retriever.search(q_vec, top_k)
            docs = rerank(query, docs)
            context += "\n".join(docs)

        if img_description and mode != "text":
            context += "\n" + img_description

        context += "\n" + st.session_state.memory.context()

        if not context.strip():
            st.warning("No context available. Upload document or image.")
            st.stop()

        answer = generate(query, context)
        st.session_state.memory.add(query, answer)

        st.markdown("### Answer")
        st.write(answer)

        st.caption(f"Latency: {elapsed(start)} sec")

    except Exception as e:
        st.error(f"Query failed: {e}")



st.markdown("---")
st.caption("Enterprise Multimodal RAG • Text + Vision")
