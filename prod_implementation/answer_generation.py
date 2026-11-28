import re
import pickle

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json
import sqlite3
from langchain_chroma import Chroma
from pathlib import Path

# ===========================================================================
# Configuration
# ===========================================================================
text_embedding_model = "BAAI/bge-large-en-v1.5"

BASE_DIR = Path(__file__).resolve().parent.parent
database_dir = "databases"
child_vector_db_name = "PROD_child_chunks_vector_db"
child_vector_db_path = BASE_DIR / database_dir / child_vector_db_name
parent_non_vector_db_name = "PROD_parent_records.sqlite3"
parent_non_vector_db_path = BASE_DIR / database_dir / parent_non_vector_db_name
bm25_index_path = BASE_DIR / database_dir / "PROD_bm25_index.pkl"

# ===========================================================================
# Variable Declaration
# ===========================================================================
text_embedding_function = HuggingFaceEmbeddings(
    model_name = text_embedding_model
)

child_vector_store = Chroma(
    persist_directory=str(child_vector_db_path),
    embedding_function=text_embedding_function,
    collection_name="child_chunks"
)

conn = sqlite3.connect(parent_non_vector_db_path)

# ===========================================================================
# BM25 Index Loading
# ===========================================================================
with open(bm25_index_path, "rb") as file:
    bm25_data = pickle.load(file)

bm25_index = bm25_data["bm25_index"]
bm25_documents = bm25_data["bm25_documents"]

# ===========================================================================
# Helper Functions
# ===========================================================================
def get_child_id(doc: Document):
    child_id = doc.metadata.get("child_id")
    
    # Typically BM25 Indexes will have child_id (i.e. parent_id::child_index)
    if child_id:
        return child_id
    else:
    # Typically Semantic Searches will not have child_id. So we create new parent_id::child_index
        return f"{doc.metadata.get("parent_id")}::{doc.metadata.get("child_index")}"
    

def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\b[\w-]+\b", text.lower())

def bm25_index_retrieval(user_prompt: str, k: int):
    tokenized_text = tokenize_text(user_prompt)
    token_scores = bm25_index.get_scores(tokenized_text)
    top_k_index = token_scores.argsort()[::-1][:k]
    bm25_chunks = [bm25_documents[index] for index in top_k_index if token_scores[index] > 0]
    return bm25_chunks

def reciprocal_rank_fusion(semantic_results: list[Document], bm25_results: list[Document], rrf_k: int = 60):
    retrieved_results = [semantic_results, bm25_results]
    
    fused_scores = {}
    child_document_map = {}
    for results in retrieved_results:
        for rank, doc in enumerate(results, start=1):
            child_id = get_child_id(doc)
            rrf_k = 1 / (rrf_k + rank)
            fused_scores["child_id"] = (fused_scores.get(child_id,0) + rrf_k)
            child_document_map[child_id] = doc
    
    # Sort child IDs based on their combined RRF scores
    ranked_child_ids = sorted(
        fused_scores,
        key=lambda child_id: fused_scores[child_id],
        reverse=True
    )
    fused_results = [child_document_map[child_id] for child_id in ranked_child_ids]
    return fused_results

def parent_child_retriever(user_prompt: str, child_k: int, parent_k: int):
    """
    user_prompt: query provided by the user
    child_k: Number of chunks to be extracted from Vector Store
    parent_k: Fetch Top K number of unique parent_ids corresponding to the child_chunks extracted

    Output:
    Returns list of Top K (parent_k) parent records corresponding to k (child_k) child _chunks
    """
    retrieved_child_chunks = child_vector_store.similarity_search(
        query=user_prompt,
        k=child_k)
    
    # Extract unique parent IDs while preserving rank
    parent_ids = []
    for child in retrieved_child_chunks:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id not in parent_ids:
            parent_ids.append(parent_id)

    parent_records = []
    for parent_id in parent_ids:
        parent_page_content, parent_metadata = conn.execute(
            """
            SELECT page_content, metadata
            from parent_documents
            where parent_id = ?
            """,
            (parent_id,)
        ).fetchone()

        parent_records.append(Document(
            page_content=parent_page_content,
            metadata = json.loads(parent_metadata)
        ))

        if len(parent_records) == parent_k:
            break
    return parent_records

