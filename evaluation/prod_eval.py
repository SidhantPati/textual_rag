import math

# from langchain.messages import SystemMessage, HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from prod_implementation.answer_generation import parent_child_retriever
# from answer_generation import retrieved_context_from_vector_db,llm_response, system_prompt
from evaluation.test1_obj import load_test_objects

gemini_model = "gemini-3.5-flash-lite"

#==============================================================================#
#                       FUNCTIONS TO EVALUATE RETRIEVAL                        #
#==============================================================================#

class RetrieveEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")

def calculate_mrr(keyword: str, retrieved_docs: list, k: int):
    """
    MRR focuses on how high the first relevant result appears in a ranked list
    if keyword presentin doc content, return 1/rank or else 0
    """
    keyword_lower = keyword.lower().strip()
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
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
    retrieved_docs = parent_child_retriever(test.question,k=3)
    
    # calculate mrr
    mrr_score_list = [calculate_mrr(keyword, retrieved_docs, k) for keyword in test.keywords]
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
    
def evaluate_all_retrieval(k):
    test_objects = load_test_objects()
    total_tests = len(test_objects)
    all_results = []
    for index, test in enumerate(test_objects):
        result = evaluate_retrieval(test, k)
        all_results.append(result)
        progress = (index+1)/total_tests
        print(f"🔁Progress: {progress*100:.2f}% ---> mrr: {result.mrr:.4f} | ndcg: {result.ndcg:.4f} | keyword_coverage: {result.keyword_coverage:.2f} | Q: {test.question}")
        # yield test, result, progress
    
    # Calculate overall metrics across all test questions
    overall_mrr = (
        sum(r.mrr for r in all_results) / total_tests
        if total_tests else 0.0
    )

    overall_ndcg = (
        sum(r.ndcg for r in all_results) / total_tests
        if total_tests else 0.0
    )

    overall_coverage = (
        sum(r.keyword_coverage for r in all_results) / total_tests
        if total_tests else 0.0
    )

    print("\n\n========== OVERALL RETRIEVAL EVALUATION ==========")
    print(f"Total test questions: {total_tests}")
    print(f"Overall MRR@{k}: {overall_mrr:.4f}")
    print(f"Overall NDCG@{k}: {overall_ndcg:.4f}")
    print(f"Overall Keyword Coverage: {overall_coverage:.2f}%")

    return 0

#===========================================================================#
#                       FUNCTIONS TO EVALUATE ANSWER                        #
#===========================================================================#

# class AnswerEval(BaseModel):
#     feedback: str = Field(description="Feedback on the answer quality by comparing with the reference answer and evaluating based on the retrieved context")
#     accuracy: float = Field(description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3.")
#     completeness: float = Field(description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included.")
#     relevance: float = Field(description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information.")

# def evaluate_answer(test):
#     response, retrieved_context = llm_response(test.question,[])
#     # LLM judge prompt
#     system_prompt = """
#     You are an expert evaluator assessing the quality of answers. 
#     Evaluate the generated answer by comparing it to the reference answer. Only give 
#     5/5 scores for perfect answers.
#     """

#     human_prompt = f"""
#     Question:
#     {test.question}

#     Generated Answer:
#     {response}

#     Reference Answer:
#     {test.reference_answer}

#     Please evaluate the generated answer on three dimensions:
#     1. Accuracy: How factually correct is it compared to the reference answer? 
#     Only give 5/5 scores for perfect answers.
#     2. Completeness: How thoroughly does it address all aspects of the question, 
#     covering all the information from the reference answer?
#     3. Relevance: How well does it directly answer the specific question asked, 
#     giving no additional information?

#     Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each 
#     dimension. If the answer is wrong, then the accuracy score must be 1.
#     """

#     llm_judge_messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=human_prompt)
#     ]

#     llm = ChatGoogleGenerativeAI(model = gemini_model).with_structured_output(AnswerEval)
#     llm_judge_eval = llm.invoke(llm_judge_messages)

#     return llm_judge_eval

# def evaluate_all_answers():
#     test_objects = load_test_objects()
#     total_tests = len(test_objects)
#     all_results = []
#     for index, test in enumerate(test_objects):
#         progress = (index+1)/total_tests
#         result = evaluate_answer(test)
#         all_results.append(result)
#         print(f"🔁Progress: {progress*100:.2f}% ---> Accuracy: {result.accuracy:.2f} | Completeness: {result.completeness:.2f} | Relevance: {result.relevance:.2f} | Q: {test.question}")
    
#     # Calculate overall metrics across all test questions
#     overall_accuracy = (
#         sum(r.accuracy for r in all_results) / total_tests
#         if total_tests else 0.0
#     )

#     overall_completeness = (
#         sum(r.completeness for r in all_results) / total_tests
#         if total_tests else 0.0
#     )

#     overall_relevance = (
#         sum(r.relevance for r in all_results) / total_tests
#         if total_tests else 0.0
#     )

#     print("\n\n========== OVERALL ANSWER EVALUATION ==========")
#     print(f"Total test questions: {total_tests}")
#     print(f"Overall Accuracy: {overall_accuracy:.2f}")
#     print(f"Overall Completeness: {overall_completeness:.2f}")
#     print(f"Overall Relevance: {overall_relevance:.2f}")

#     return 0

if __name__ == "__main__":
    evaluate_all_retrieval(k=3)
    # evaluate_all_answers()
    