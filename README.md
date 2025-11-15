# AmbedkarGPT – AI Intern Task (KalpIT Pvt Ltd)

This repository contains my submission for the **AI Intern – Phase 1 Core Skills Evaluation**.

The goal of this project is to build a **local RAG-based Q&A system** using:

- **Python 3.8+**
- **LangChain**
- **ChromaDB (local vector store)**
- **HuggingFace MiniLM embeddings**
- **Ollama + Mistral 7B (local LLM)**

Everything runs **fully offline**, with **no API keys and no external dependencies**.

---

## 🚀 Features

- Loads and processes Dr. B.R. Ambedkar's speech.
- Splits text into chunks for better retrieval.
- Creates sentence embeddings using MiniLM.
- Stores vectors locally using ChromaDB.
- Retrieves relevant context based on user questions.
- Uses Mistral-7B via Ollama to generate answers.

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
