import math

from pydantic import BaseModel, Field
from answer_generation import retrieved_context_from_vector_db
from evaluation.test1_obj import load_test_objects

class RetrieveEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")

def calculate_mrr(keyword: str, retrieved_docs: list):
    """
    MRR focuses on how high the first relevant result appears in a ranked list
    if keyword presentin doc content, return 1/rank or else 0
    """
    keyword_lower = keyword.lower().strip()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0/rank
    return 0.0

def calculate_dcg(relevances,k):
    dcg=0.0
    for i in range(min(k, len(relevances))):
        dcg = dcg + relevances[i]/math.log2(i+2)
    return dcg

def calculate_ndcg(keyword, retrieved_docs, k):
    """
    NDCG evaluates ranking quality of retrieved chunks based on keyword match
    if keyword presentin doc content, return 1/rank or else 0
    """
    keyword_lower = keyword.lower().strip()
    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances,k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(test: BaseModel, k: int = 3):
    # Retrieve all langchian docs based on the question
    retrieved_docs = retrieved_context_from_vector_db(test.question)
    
    # calculate mrr
    mrr_score_list = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_score_list)/len(mrr_score_list) if mrr_score_list else 0.0

    # calculate ndcg
    ndcg_score_list = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_score_list)/len(ndcg_score_list) if ndcg_score_list else 0.0

    # Calculate keyword coverage
    keywords_found = sum(1 for score in mrr_score_list if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0


    return RetrieveEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )
    
def evaluate_all_retrieval():
    test_objects = load_test_objects()
    total_tests = len(test_objects)
    for index, test in enumerate(test_objects):
        result = evaluate_retrieval(test)
        progress = (index+1)/total_tests
        print(f"mrr: {result.mrr} | ndcg: {result.ndcg} | keyword_coverage: {result.keyword_coverage} | Q: {test.question}")
        # yield test, result, progress
        return 0

if __name__ == "__main__":
    evaluate_all_retrieval()
    