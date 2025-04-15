import re
import fitz
import nltk
import faiss
import pickle
import logging
import requests
from pathlib import Path
from typing import List
import hashlib
import re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

nltk.download('punkt')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_pdf(url: str, dest_path: Path):
    response = requests.get(url)
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    logger.info(f"PDF downloaded to {dest_path}")

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    logger.debug(f"Extracted text:\n{text}")
    return text

def split_by_headers(text: str) -> List[str]:
    pattern = r'(?=^[A-Z][A-Z\s]+$)'
    sections = re.split(pattern, text, flags=re.MULTILINE)
    return [section.strip() for section in sections if section.strip()]

def sentence_tokenize_sections(sections: List[str]) -> List[str]:
    sentences = []
    for section in sections:
        sentences.extend(nltk.sent_tokenize(section))
    return sentences

def clean_extracted_text(text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def smart_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    sections = split_by_headers(text)
    sentences = sentence_tokenize_sections(sections)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(" ".join(sentences))
    chunks = [clean_extracted_text(chunk) for chunk in chunks]
    logger.debug("Chunks:")
    logger.debug("\n\n---\n\n".join(chunks))
    return chunks

def embed_chunks(chunks: List[str], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, embeddings

def build_faiss_index(embeddings) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatL2, chunks: List[str], index_path: Path):
    faiss.write_index(index, str(index_path.with_suffix(".index")))
    with open(index_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(chunks, f)

def create_folders(*paths: Path):
    for path in paths:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")


def ingest_documents(pdf_url: str, index_path: Path, cache_dir: Path = Path("data/cache")):
    create_folders(index_path.parent, cache_dir)
    filename_hash = hashlib.md5(str(pdf_url).encode()).hexdigest()[:8]
    cached_pdf_path = cache_dir / f"rulebook_{filename_hash}.pdf"

    if not cached_pdf_path.exists():
        logger.info(f"Downloading PDF from {pdf_url}...")
        download_pdf(pdf_url, cached_pdf_path)
    else:
        logger.info(f"Using cached PDF at {cached_pdf_path}")

    logger.info("Extracting text...")
    text = extract_text_from_pdf(cached_pdf_path)

    logger.info("Chunking text...")
    chunks = smart_chunk(text)

    logger.info("Embedding chunks...")
    _, embeddings = embed_chunks(chunks)

    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings)

    logger.info(f"Saving index to {index_path}...")
    save_index(index, chunks, index_path)

    logger.info("Ingestion complete.")

