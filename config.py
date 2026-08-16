import os

WORLD_NAME = os.environ.get("WORLD_NAME", "Ashvale")
SETTING_NAME = os.environ.get("SETTING_NAME", "the kingdom of Ashvale")
INTERFACE_TITLE = os.environ.get("INTERFACE_TITLE", "NARRATIVE DIALOGUE INTERFACE")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
DIALOGUE_CSV = os.path.join(DATA_DIR, "dialogue.csv")
KNOWLEDGE_BASE_JSON = os.path.join(DATA_DIR, "knowledge_base.json")
CURATED_FACTS_JSON = os.path.join(DATA_DIR, "curated_facts.json")
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "./vector_db")
NPCS_JSON = os.path.join(DATA_DIR, "npcs.json")
EVAL_RAG_JSON = os.path.join(DATA_DIR, "eval_rag.json")
EVAL_LLM_JSON = os.path.join(DATA_DIR, "eval_llm.json")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL_HYDE = os.environ.get("MODEL_HYDE", "gemini-3.1-flash-lite")
MODEL_DIALOGUE = os.environ.get("MODEL_DIALOGUE", "claude-haiku-4-5")
MODEL_JUDGE = os.environ.get("MODEL_JUDGE", "claude-sonnet-4-6")
MODEL_CONDENSE = os.environ.get("MODEL_CONDENSE", "claude-haiku-4-5")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "npc_knowledge")

def get_device():
    device = os.environ.get("DEVICE", "auto")
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
