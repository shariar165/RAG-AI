# student-ai-rag

under construnction

An AI-powered backend system designed to help Bangladeshi students (Classes 6–12) get accurate, textbook-based answers in Bangla.  
This project uses Retrieval-Augmented Generation (RAG) to answer students' questions using local textbook content.

---

## 🚀 Features

- ✅ Accepts questions in Bangla
- ✅ Retrieves context from textbooks
- ✅ Uses LLM (DeepSeek or compatible) to generate answers
- ✅ Custom model tuning possible for BD curriculum
- ✅ Supports translation pipeline (English ↔ Bangla) if needed
- 🛠 Built with open-source, low-resource tools

---

## 🧠 Tech Stack

- **Python 3.10+**
- **LangChain**
- **ChromaDB**
- **HuggingFace Transformers**
- **SentencePiece**
//- **Argos Translate** *(for Bangla ↔ English)* 
- **RAG Pipeline**

---

## 📁 Project Structure

student-ai-rag/
├── chroma_env/               # Python virtual environment (should not be pushed to GitHub)
├── data/
│   ├── books/                # Source files like PDFs or text documents
│   │   └── History & Social Science.pdf
│   ├── images/               # Image files for OCR processing
│   └── web/
│       └── urls.txt          # List of web URLs for scraping
├── db/
│   ├── image_embeddings/     # Chroma DB from OCR-extracted text
│   ├── pdf_embeddings/       # Chroma DB from PDFs
│   └── web_embeddings/       # Chroma DB from scraped web content
├── tools/
│   ├── ingest_pdf.py         # Tool to process and embed PDFs
│   ├── ocr_and_ingest.py     # Tool to extract text from images using OCR and embed
│   └── scrape_and_ingest.py  # Tool to scrape web URLs and embed text
├── utils/
│   └── text_cleaner.py       # Utility script for cleaning extracted text
├── main.py                   # Main script to input questions and return answers
├── requirements.txt          # Required Python libraries
└── README.md                 # Project overview and instructions

