from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# ========================= Configuration ========================= #
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "knowledge-base"
print(file_path)

# ========================= Load Knowledge Base ========================= #
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

# ========================= KB Chunking ========================= #
parent_chunk_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True)

parent_chunk = parent_chunk_text_splitter.split_documents(knowledge_base)

# chlid_chunk_text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
#     add_start_index=True
# )

# for index, chunk in enumerate(parent_chunk):
