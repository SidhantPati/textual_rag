# 🚀 RAG Chat Application

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB, and Google Gemini based on fictional insurance company, InsureLLM (inspired from Ed Donner's "AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents" course on Udemy).

## ✅ Features

- **RAG Pipeline**: Retrieves relevant document chunks to augment LLM responses
- **ChromaDB**: Vector database for storing and searching document embeddings
- **Google Generative AI**: Uses Gemini for both text embeddings and chat responses
- **Chat History**: Maintains conversation history for context-aware responses
- **Environment Management**: Secure API key management using `python-dotenv`

## ✅ Usage

The main entry point is `main.py`, which includes gradio based UI which helps to drive conversationa and also check the retrieved_context docs for reference. You can run it directly:

```bash
python main.py
```

To drive 1:1 LLM call, use `answer_generation.py`. You can run it directly:

```bash
python answer_generation.py
```

To re-factor embedding ingestion pipeline into chroma vector db, use use `vector_ingestion.py`. You can run it directly:

```bash
python vector_ingestion.py
```

## ✅ Evaluation Metrics

Evaluation script is `evalaution/test1_eval.py`. You can run it directly:

```bash
python -m evaluation.test1_eval
```

### Answer Module (LLM-as-a-Judge):

Based on 150 test questions, the base Answer model registered the following metrics, as evaluated via LLM-as-a-Judge framework.

| Metric | Score |
|---|---:|
| Overall Accuracy | 4.57 / 5.00 |
| Overall Completeness | 4.38 / 5.00 |
| Overall Relevance | 4.25 / 5.00 |


### Retrieval Module:

Based on 150 test questions, the base retrieval model registered the following metrics.

| Metric | Score |
|---|---:|
| Overall MRR@3 | 0.8669 |
| Overall NDCG@3 | 0.8739 |
| Overall Keyword Coverage | 91.41% |

## ✅ Project Structure

```
textual_rag/
├── .env                    # Environment variables (not in git)
├── .venv/                  # Python virtual environment
├── main.py                 # Main RAG application (Chat UI)
├── answer_generation.py    # RAG Response Generation Module
├── vector_ingestion.py     # Document Ingestion into ChromaDB Vector db
├── evaluation/             # Conduct evaluation
│   ├── test1.jsonl
│   ├── test1_obj.py
│   └── test1_eval.py
├── knowledge-base/         # Document source files
│   ├── company
│       ├── about.md
│       ├── careers.md
│       ...
│   ├── contracts
│   ├── employees
│   └── products
└── README.md               # Project documentation
```

## ✅ Getting Started

### Prerequisites

- Python 3.9+
- ChromaDB
- Google Gemini API Key

### Installation & Execution
1. Clone the repository and install requirements.
2. Create a `.env` file in the root directory and add your API keys.
3. **Build the Vector DB:**
   Run `python vector_ingestion.py` to trigger the LLM-driven semantic chunking and store embedding into the local ChromaDB.
4. **Run Evaluations:**
   Run `python evaluation/test1_eval.py` to launch the cli based evaluation and run the comprehensive test suite.

### Troubleshooting

- **API Key Issues**: Ensure `GOOGLE_API_KEY` is set in your `.env` file and that you have access to the specified Gemini model.
- **ChromaDB Not Found**: Run `vector_ingestion.py` first to populate the vector database.
- **Module Not Found**: Ensure you've activated the virtual environment: `source .venv/bin/activate`.
