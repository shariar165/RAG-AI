import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

PDF_PATH = "../data/books/your_book.pdf"
CHROMA_DB_DIR = "../db/pdf_embeddings"

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def save_to_chroma(docs, persist_dir):
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print(f"[✓] Stored {len(docs)} PDF chunks to {persist_dir}")

if __name__ == "__main__":
    docs = load_and_split_pdf(PDF_PATH)
    chunks = chunk_docs(docs)
    save_to_chroma(chunks, CHROMA_DB_DIR)
