# Narrative RAG Dialogue System

A retrieval-augmented generation (RAG) pipeline for interactive NPC dialogue. Raw source text is processed into a structured lore database, embedded into a vector store, and retrieved at runtime to ground LLM-generated character responses.

## Pipeline

```
dialogue.csv
    → extract.py (spaCy NER)
    → knowledge_base.json
    → llm_process.py (Gemini distillation)
    → curated_facts.json
    → embed.py (ChromaDB + sentence embeddings)
    → vector_db/
    → generate.py / conversation.py (RAG + Claude dialogue)
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

2. Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Required keys:
- `GEMINI_API_KEY` — HyDE generation and fact distillation
- `ANTHROPIC_API_KEY` — dialogue generation, history condensation, and evaluation

3. Build the vector database:

```bash
python embed.py
```

The repository includes a pre-built demo world (`data/`) with sample dialogue, lore, and NPC definitions. To rebuild from scratch:

```bash
python extract.py
python llm_process.py
python embed.py
```

## Usage

Interactive dialogue:

```bash
python conversation.py
```

Evaluate retrieval quality:

```bash
python evaluate_rag.py
```

Evaluate end-to-end dialogue quality (requires API keys):

```bash
python evaluate_llm.py
```

## Customisation

All paths and world settings are configured via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `WORLD_NAME` | Ashvale | Name used in archivist and judge prompts |
| `SETTING_NAME` | the kingdom of Ashvale | Setting referenced in dialogue prompts |
| `DATA_DIR` | `./data` | Directory for all data files |
| `VECTOR_DB_PATH` | `./vector_db` | ChromaDB storage location |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

To use your own fictional world, replace the files in `data/`:

- `dialogue.csv` — source text (one line per row, `content` column)
- `npcs.json` — NPC personas, scenes, and few-shot examples
- `eval_rag.json` / `eval_llm.json` — evaluation fixtures
- Run the pipeline to produce `knowledge_base.json` and `curated_facts.json`, or author them directly

## Hardware

Embedding and reranking models use GPU acceleration when available. Set `DEVICE=cpu` if no CUDA device is present.
