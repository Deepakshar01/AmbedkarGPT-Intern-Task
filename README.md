# AmbedkarGPT – AI Intern Assignment (KalpIT)

This project is my submission for the AI Intern assignment at **KalpIT**.  
The goal of the task was to build a **Retrieval-Augmented Generation (RAG)**-based Question Answering system that answers queries using the provided *speech.txt* document.

I implemented the complete pipeline including document loading, text chunking, vector embeddings, vector database creation, and LLM-based answer generation using an Ollama model.

---

## 🚀 Features

- ✔ Loads the provided **speech.txt** file  
- ✔ Splits the text into meaningful chunks  
- ✔ Generates vector embeddings using **SentenceTransformers (MiniLM-L6-v2)**  
- ✔ Stores them in an in-memory **FAISS** vector database  
- ✔ Uses a lightweight **Ollama model** (`tinyllama`) for inference  
- ✔ Answers any question only using information from the speech  
- ✔ Shows retrieved source passages for transparency  
- ✔ Clean and minimalistic console-based UI

---

## 🧠 Tech Stack

- **Python 3.10**
- **Ollama (tinyllama model)**
- **LangChain**
- **Sentence Transformers**
- **FAISS**
- **Streamlit** (optional UI)

---

## 📂 Project Structure

project/
│── main.py
│── app.py (optional Streamlit app)
│── requirements.txt
│── speech.txt
│── README.md
│── venv/

yaml
Copy code

---

## ▶️ How to Run the Project

### **1. Create Virtual Environment**
```bash
python -m venv venv
2. Activate Environment
bash
Copy code
./venv/Scripts/activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Start Ollama
Download any lightweight model such as:

bash
Copy code
ollama pull tinyllama
5. Run the App
bash
Copy code
python main.py
You will see:

diff
Copy code
=== AmbedkarGPT Ready ===
Ask a question:
💡 Example Questions
“Who is the real enemy according to the speech?”

“What message does the speaker want to convey?”

“What are the biggest challenges discussed?”

📘 Sample Output
vbnet
Copy code
Answer:
 According to the speech, the enemy is the belief in the
 sanctity of the castes…

Sources:
 - speech.txt | score: 0.25
 - speech.txt | score: 0.10
🎯 What I Learned
How RAG pipelines work end-to-end

Using LangChain with local LLMs

Choosing lightweight models based on system RAM

Debugging dependency and environment issues

Creating clean GitHub project documentation

📎 Assignment Requirements Covered
 Use of LLM

 Use of embeddings

 Use of RAG

 Working question answering system

 Clean code + comments

 GitHub repository link ready for submission

✨ Author
Deepak Sharma
AI/ML & Python Enthusiast
B.Sc Computer Science (2023–2027)

