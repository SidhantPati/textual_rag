import json
import shutil
import sqlite3

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# ===========================================================================
# Configuration
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "knowledge-base"
parent_records = [] # Store list of updated parent chunk information to be stored in non-vector db
child_records = [] # Store list of langchain_docs (i.e. child chunks) to be store in vector db

text_embedding_model = "BAAI/bge-large-en-v1.5"
database_dir = "databases" # New directory
child_vector_db_name = "PROD_child_chunks_vector_db"
child_vector_db_path = BASE_DIR / database_dir / child_vector_db_name
parent_non_vector_db_name = "PROD_parent_records.sqlite3"
parent_non_vector_db_path = BASE_DIR / database_dir / parent_non_vector_db_name

delete_existing_child_vector_store = True

# ===========================================================================
# Load Knowledge Base
# ===========================================================================
loader = DirectoryLoader(
    path=file_path,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'})

knowledge_base = loader.load()

for doc in knowledge_base:
    doc.metadata["source_type"] = Path(doc.metadata["source"]).parent.name
    doc.metadata["file_name"] = Path(doc.metadata["source"]).stem

print(f"\n✅Success!! {len(knowledge_base)} Documents loaded")

# ===========================================================================
# Parent Chunking
# ===========================================================================
parent_chunk_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True)

parent_chunks = parent_chunk_text_splitter.split_documents(knowledge_base)
print(f"\n✅Success!! {len(parent_chunks)} Parent Chunks created")

# ===========================================================================
# Child Chunking
# ===========================================================================
chlid_chunk_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)

for parent_index, parent_chunk in enumerate(parent_chunks):
    parent_source =  parent_chunk.metadata["source"]
    parent_id = f"{parent_source}::{parent_index}"
    
    parent_metadata = parent_chunk.metadata.copy()
    parent_metadata["parent_id"] = parent_id
    parent_metadata["parent_index"] = parent_index

    parent_records.append({
        "parent_id":parent_id,
        "page_content": parent_chunk.page_content,
        "metadata":parent_metadata
    })

    child_chunks = chlid_chunk_text_splitter.split_documents([parent_chunk])
    for child_index, child_chunk in enumerate(child_chunks):
        child_chunk.metadata["parent_id"] = parent_id
        child_chunk.metadata["child_index"] = child_index
        child_records.append(child_chunk)
print(f"\n✅Success!! {len(child_chunks)} Child Chunks created")

# ===========================================================================
# Store Child Chunks in Vector DB
# ===========================================================================

if delete_existing_child_vector_store and child_vector_db_path.exists():
    print(f"🗑️ Removing existing database at: {child_vector_db_path}")
    shutil.rmtree(child_vector_db_path)
print("⏳ Building new vector database...")
# Unique identifier for the vector db
child_vector_db_ids = [f"{child_doc.metadata["parent_id"]}::{child_doc.metadata["child_index"]}" for child_doc in child_records]
embeddings = HuggingFaceEmbeddings(
    model_name = text_embedding_model
)
child_vector_store = Chroma.from_documents(
    documents=child_records,
    embedding=embeddings,
    ids=child_vector_db_ids,
    collection_name="child_chunks",
    persist_directory=child_vector_db_path
)
print(f"\n✅Success!! Child Vector Store created and {len(child_chunks)} ingested")

# ===========================================================================
# Store Parent Chunks in Non-Vector DB (SQLite)
# ===========================================================================
parent_non_vector_db_path.parent.mkdir(parents=True, exist_ok=True)

# DEFINE SQLite non-vector store for parent docs
conn = sqlite3.connect(parent_non_vector_db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS parent_documents (
        parent_id TEXT PRIMARY KEY,
        page_content TEXT NOT NULL,
        metadata TEXT NOT NULL
    )
""")
conn.commit()

# Clear existing parent documents
conn.execute("DELETE FROM parent_documents")
conn.commit()

# Insert newly generated parent documents
conn.executemany(
    """
    INSERT INTO parent_documents
    (parent_id, page_content, metadata)
    VALUES (?, ?, ?)
    """,
    [
        (
            record["parent_id"],
            record["page_content"],
            json.dumps(record["metadata"])
        )
        for record in parent_records
    ]
)
conn.commit()
print(f"\n✅Success!! Parent SQLite Store created and {len(parent_records)} ingested")
conn.close()