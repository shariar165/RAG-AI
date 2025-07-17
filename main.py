import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
from utils.text_cleaner import clean_text


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        text = clean_text(text)  # <-- Clean the text here
        pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text, max_tokens=200):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length > max_tokens:
            chunks.append('.'.join(current_chunk).strip())
            current_chunk = [sentence]
            current_length = token_length
        else:
            current_chunk.append(sentence)
            current_length += token_length
    if current_chunk:
        chunks.append('.'.join(current_chunk).strip())
    return chunks

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks)
    return embeddings

def store_in_vector_db(chunks, embeddings, page_mapping):
    client = chromadb.Client(Settings(persist_directory="db/pdf_embeddings"))
    collection = client.get_or_create_collection(name="class9_history")
    for i, (chunk, embedding, page_num) in enumerate(zip(chunks, embeddings, page_mapping)):
        collection.add(
            documents=[chunk],
            ids=[f"chunk-{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"page": page_num}]
        )
    client.persist()  # Save the DB to disk
    return collection


def retrieve_relevant_chunks(collection, query, model_name="all-MiniLM-L6-v2", top_k=3):
    embedder = SentenceTransformer(model_name)
    query_vector = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(documents, metadatas))

def ask_ollama_llm(context_chunks, question, model="deepseek-r1:1.5b"):
    # Add page numbers if included in the chunk text
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant answering questions from a textbook. The user is asking a question based on the textbook 'History & Social Science Class 9 - English Version'.

Use only the context below to answer. If the answer is not found, say "Not available in the book."

Include page numbers from the context in your answer (e.g., "This topic is discussed on Page 88").

Format:
Answer:
<Your direct answer>

Pages: <List of page numbers used>

Context:
{context}

Question: {question}
Answer:
"""

    response = ollama.chat(model=model, messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']


def main():
    pdf_path = "History & Social Science Class 9 English version.pdf"

    # 1. Extract
    pages = extract_text_from_pdf(pdf_path)

    # 2. Chunk with page tracking
    chunks, page_mapping = [], []
    for page in pages:
        c = chunk_text(page["text"])
        chunks.extend(c)
        page_mapping.extend([page["page"]] * len(c))

    # 3. Embed
    embeddings = embed_chunks(chunks)

    # 4. Store in vector DB
    collection = store_in_vector_db(chunks, embeddings, page_mapping)

    # 5. Ask
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        results = retrieve_relevant_chunks(collection, query)
        answer = ask_ollama_llm(results, query)
        print("\n🔎 Answer:\n", answer)

if __name__ == "__main__":
    main()
