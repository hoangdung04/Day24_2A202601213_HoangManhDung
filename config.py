"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- LLM Configuration ---
if GROQ_API_KEY:
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
else:
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1" if OPENAI_API_KEY else "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab24_production"
NAIVE_COLLECTION = "naive_baseline"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

# --- LLM Judge ---
JUDGE_MODEL = "gpt-4o-mini"

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer


def get_llm_client():
    """Trả về OpenAI client (Groq hoặc OpenAI) hoặc None nếu không có API key hợp lệ."""
    groq_key = os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY
    openai_key = os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY

    try:
        from openai import OpenAI
        if groq_key:
            base_url = os.getenv("LLM_BASE_URL") or LLM_BASE_URL or "https://api.groq.com/openai/v1"
            return OpenAI(api_key=groq_key, base_url=base_url)
        elif openai_key and openai_key.strip():
            base_url = os.getenv("LLM_BASE_URL") or LLM_BASE_URL
            if base_url and "groq" not in base_url:
                return OpenAI(api_key=openai_key, base_url=base_url)
            return OpenAI(api_key=openai_key)
    except Exception:
        return None

    return None
