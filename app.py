import os
import numpy as np
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# ---------- Core RAG Logic (same brain as your CLI version) ----------

def load_speech(path: str = "speech.txt") -> str:
    """Load the raw text of the speech from a file."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """Simple character-based text splitter with overlap."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # overlap between chunks

    return chunks


class SimpleVectorStore:
    """
    A tiny in-memory vector store:
    - Stores embeddings + texts
    - Does cosine similarity search
    """

    def __init__(self, texts, metadatas, embedding_model: HuggingFaceEmbeddings):
        self.texts = texts
        self.metadatas = metadatas
        self.embedding_model = embedding_model
        # Pre-compute embeddings for all chunks
        self.embeddings = np.array(self.embedding_model.embed_documents(texts))

    def similarity_search(self, query: str, k: int = 3):
        """Return top-k most similar chunks to the query."""
        query_emb = np.array(self.embedding_model.embed_query(query))

        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(query_emb) + 1e-8)
        sims = (self.embeddings @ query_emb) / (norms + 1e-8)

        top_k_idx = sims.argsort()[-k:][::-1]

        docs = []
        for idx in top_k_idx:
            docs.append(
                {
                    "page_content": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(sims[idx]),
                }
            )
        return docs


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    # tinyllama works on low-RAM systems
    return Ollama(model="tinyllama")


@st.cache_resource
def build_vector_store(speech_text: str):
    """Create embeddings + simple in-memory vector store, cached for performance."""
    if not speech_text.strip():
        return None

    chunks = split_text(speech_text, chunk_size=300, overlap=50)
    embeddings = get_embeddings()
    metadatas = [{"source": "speech.txt"} for _ in chunks]

    store = SimpleVectorStore(
        texts=chunks,
        metadatas=metadatas,
        embedding_model=embeddings,
    )
    return store


def answer_question(question: str, vectordb: SimpleVectorStore, llm: Ollama):
    """Retrieve relevant chunks and ask the LLM to answer using only that context."""
    docs = vectordb.similarity_search(question, k=3)

    context = "\n\n".join(doc["page_content"] for doc in docs)

    prompt = (
        "You are AmbedkarGPT, a helpful assistant that answers questions ONLY using the given speech.\n\n"
        "Speech excerpt:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in 2–4 sentences based strictly on the speech above. "
        "If the answer is not in the text, say: "
        "\"The speech does not give enough information to answer this question.\"\n"
    )

    answer = llm.invoke(prompt)
    return answer, docs


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="AmbedkarGPT – RAG over Speech", page_icon="📜")
    st.title("📜 AmbedkarGPT – RAG Q&A over a Speech")
    st.write(
        "Ask questions about Dr. B. R. Ambedkar's speech. "
        "The model answers **only** using the content of the uploaded / loaded speech."
    )

    with st.sidebar:
        st.header("⚙️ Settings")

        st.markdown("**Speech Source**")
        uploaded_file = st.file_uploader(
            "Upload a `.txt` file (optional)", type=["txt"], help="If not provided, speech.txt in the project folder will be used."
        )

        if uploaded_file is not None:
            speech_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success("Using uploaded speech file.")
        else:
            speech_text = load_speech("speech.txt")
            if speech_text:
                st.info("Using local `speech.txt` from project folder.")
            else:
                st.error("No speech found. Please upload a .txt file or add speech.txt to the project folder.")
                st.stop()

        st.markdown("---")
        st.caption("Embeddings model: `all-MiniLM-L6-v2`")
        st.caption("LLM model (Ollama): `tinyllama`")

    # Build / get vector store
    with st.spinner("Indexing speech (building embeddings)..."):
        vectordb = build_vector_store(speech_text)

    if vectordb is None:
        st.error("Could not build vector store. Check the speech text.")
        st.stop()

    # Chat-like interface
    st.subheader("💬 Ask a question about the speech")

    question = st.text_input(
        "Enter your question",
        placeholder="e.g. Who is the real enemy according to the speech?",
    )

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking with tinyllama + RAG..."):
            llm = get_llm()
            answer, docs = answer_question(question, vectordb, llm)

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("📚 Retrieved Chunks / Sources"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}** (score: `{doc['score']:.3f}`)")
                st.write(doc["page_content"])
                st.caption(f"Source: {doc['metadata'].get('source', 'speech.txt')}")
                st.markdown("---")


if __name__ == "__main__":
    main()
