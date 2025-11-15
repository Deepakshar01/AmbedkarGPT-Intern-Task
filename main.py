# main.py
# ------------------------------------
# AmbedkarGPT - Simple RAG Q&A System (No Chroma, custom vector store)
# Uses: LangChain embeddings + Ollama (Mistral) + custom in-memory vector store
# ------------------------------------

import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


def load_speech(path: str = "speech.txt") -> str:
    """Load the raw text of the speech from a file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path} in the current directory.")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    Very simple manual text splitter.
    Splits long text into overlapping character chunks.
    """
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

        # Cosine similarity
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


def build_vector_store(chunks):
    """Create embeddings + simple in-memory vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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

    answer = llm(prompt)
    return answer, docs


def main():
    print("\n=== AmbedkarGPT Ready ===\n")

    # 1. Load speech text
    text = load_speech("speech.txt")

    # 2. Split into chunks (manual splitter)
    chunks = split_text(text, chunk_size=300, overlap=50)

    # 3. Build *custom* vector store with embeddings
    vectordb = build_vector_store(chunks)

    # 4. Create LLM (Ollama + mistral running locally)
    llm = Ollama(model="tinyllama")

    # 5. Simple command-line Q&A loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer, docs = answer_question(query, vectordb, llm)

        print("\nAnswer:\n", answer)
        print("\nSources:")
        for doc in docs:
            print(" -", doc["metadata"].get("source", "speech.txt"), "| score:", round(doc["score"], 3))


if __name__ == "__main__":
    main()
