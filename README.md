# YouTube Transcript Chatbot

A RAG (Retrieval-Augmented Generation) application that allows users to ask questions about any YouTube video with available transcripts.

## Features

* Fetches YouTube video transcripts
* Splits transcripts into chunks
* Generates embeddings using BAAI/bge-small-en-v1.5
* Stores embeddings in FAISS vector database
* Retrieves relevant context
* Answers questions using Qwen2.5-72B-Instruct
* Streamlit-based web interface

## Tech Stack

* Python
* LangChain
* Hugging Face
* FAISS
* Streamlit
* YouTube Transcript API

## How to Run

1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt

3. Add your Hugging Face API token
4. Run:

streamlit run app.py

## Architecture

YouTube Transcript → Text Splitter → Embeddings → FAISS → Retriever → Qwen LLM → Answer
