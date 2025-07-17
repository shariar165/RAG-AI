import pytesseract
from PIL import Image
from langchain.document_loaders import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

IMAGE_PATH = "../data/images/page.png"
CHROMA_DB_DIR = "../db/image_embeddings"

def extract_text_from_image(img_path):
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    metadata = {"source": img_path}
    return Document(page_content=text, metadata=metadata)

def chunk_and_store(doc):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    db.persist()
    print(f"[✓] Stored {len(chunks)} image chunks from {doc.metadata['source']}")

if __name__ == "__main__":
    doc = extract_text_from_image(IMAGE_PATH)
    chunk_and_store(doc)
