import faiss
import pickle
import logging
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_index(index_path: Path):
    """
    Load the FAISS index and corresponding chunks from disk.
    """
    logger.info(f"Loading index from {index_path}...")
    index = faiss.read_index(str(index_path.with_suffix(".index")))
    with open(index_path.with_suffix(".pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_query(query: str, model: SentenceTransformer):
    """
    Convert the user's query into an embedding using the specified model.
    """
    logger.info("Embedding query...")
    return model.encode([query])

def retrieve_top_k(query_embedding, index, chunks: List[str], k: int = 5):
    """
    Retrieve the top k relevant chunks from the index based on the query embedding.
    """
    logger.info(f"Retrieving top {k} relevant chunks...")
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def retrieve_context(index_path: Path, query: str, model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
    """
    Retrieve relevant context chunks for a given query.
    """
    index, chunks = load_index(index_path)
    model = SentenceTransformer(model_name)
    query_embedding = embed_query(query, model)
    retrieved_chunks = retrieve_top_k(query_embedding, index, chunks, top_k)
    return retrieved_chunks
