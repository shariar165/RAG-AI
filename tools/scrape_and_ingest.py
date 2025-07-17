import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

URL = "https://en.wikipedia.org/wiki/French_Revolution"
CHROMA_DB_DIR = "../db/web_embeddings"

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    full_text = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    metadata = {"source": url}
    return Document(page_content=full_text, metadata=metadata)

def split_and_store(doc):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    db.persist()
    print(f"[✓] Stored {len(chunks)} web chunks from {doc.metadata['source']}")

if __name__ == "__main__":
    doc = scrape_website(URL)
    split_and_store(doc)
