import re
import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\b[\w-]+\b", text.lower())

def create_bm25_index(child_vector_store, bm25_index_path):
    """
    Mapping of LangChain Document object to Chroma DB attributes
    LangChain Document	            ChromaDB .get()
    page_content	                documents
    metadata	                    metadatas
    id (if assigned separately)	    ids
    """
    child_records = child_vector_store.get(
        include = ["documents","metadata"]
    )

    bm25_documents = []
    for id, doc, metadata in zip(child_records["ids"],child_records["documents"],child_records["metadata"]):
        metadata = dict(metadata or {})
        metadata["child_id"] = id
        bm25_documents.append(Document(
            page_content=doc,
            metadata=metadata
        ))

    # List of Tokenized documents
    tokenized_text_list = [tokenize_text(doc.page_content) for doc in bm25_documents]
    # BM25Okapi builds the BM25 index and calculates corpus statistics (like tf-idf, doc length etc)
    bm25_index = BM25Okapi(tokenized_text_list)

    with open(bm25_index_path,"wb") as file:
        pickle.dump(
            {
                "bm25_index":bm25_index,
                "bm25_documents":bm25_documents
            },
            file)

    return None