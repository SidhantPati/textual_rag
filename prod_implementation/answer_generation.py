import __main__
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
# Helper Functions
# ===========================================================================
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