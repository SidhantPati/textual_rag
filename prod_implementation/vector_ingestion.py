import sqlite3

from langchain_community.document_loaders import DirectoryLoader, TextLoader
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
vector_db_name = "PROD_child_chunks_vector_db"
non_vector_db_name = "PROD_parent_records.sqlite3"

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

    parent_records.apend({
        "parent_id":parent_id,
        "content": parent_chunk.page_content,
        "metadata":parent_metadata
    })

    child_chunks = chlid_chunk_text_splitter.split_documents([parent_chunk])
    for child_index, child_chunk in enumerate(child_chunks):
        child_chunk.metadata["parent_id"] = parent_id
        child_chunk.metadata["child_index"] = child_index
        child_records.append(child_chunk)
print(f"\n✅Success!! {len(child_chunks)} Child Chunks created")

# ===========================================================================
# Store Parent Records in Non-Vector DB and Child Chiunks in Vector DB
# ===========================================================================
