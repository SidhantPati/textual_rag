import __main__
from langchain_core.documents import Document
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
child_vector_store = Chroma(
    persist_directory=str(child_vector_db_path),
    embedding_function=text_embedding_model
)

conn = sqlite3.connect(parent_non_vector_db_path)

# ===========================================================================
# Helper Functions
# ===========================================================================
def parent_child_retriever(user_prompt: str, k: int):
    retrieved_child_chunks = child_vector_store.similarity_search(
        query=user_prompt,
        k=k)
    
    parent_ids = list(set([child.metadata["parent_id"] for child in retrieved_child_chunks if child]))

    parent_records = []
    for parent_id in parent_ids:
        parent_page_content, parent_metadata = conn.execute(
            """
            SELECT page_content, metadata
            from parent_documents
            where parent_id = ?
            """,
            (parent_id)
        ).fetchone()

        parent_records.append(Document(
            page_content=parent_page_content,
            metadata = json.loads(parent_metadata)
        ))
    return parent_records