import json
from pathlib import Path
from pydantic import BaseModel, Field

test_file_path = str(Path(__file__).parent / "test1.jsonl")

class TestQuestion(BaseModel):
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category_str: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")

def load_test_objects():
    test_object = []
    with open(test_file_path,"r",encoding="utf-8") as file:
        for test in file:
            test_data = json.load(test.strip())
            data_object = TestQuestion(**test_data) # convert json to pydantic object for schema validation
            test_object.append()
    return test_object