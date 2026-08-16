from generate import get_collection, retrieve_lore
from contexts import CONTEXTS, load_context
import json
from config import EVAL_RAG_JSON
import time

def rank_retrieval(retrieved_chunks, targets):
    hits = 0
    sum_precisions = 0
    found_targets = set()
    total_possible = len(targets)

    for rank_idx, doc in enumerate(retrieved_chunks):
        rank = rank_idx + 1
        doc_lower = doc.lower()
        
        new_hit_in_this_chunk = False
        for t in targets:
            if t.lower() in doc_lower and t not in found_targets:
                hits += 1
                found_targets.add(t)
                new_hit_in_this_chunk = True
        
        if new_hit_in_this_chunk:
            precision_at_k = hits / rank
            sum_precisions += precision_at_k

    ap = sum_precisions / total_possible

    return {
        "hits": hits,
        "total": total_possible,
        "ap": ap
    }

with open(EVAL_RAG_JSON, 'r', encoding='utf-8') as f:
    TEST_DATA = json.load(f)

configs = [
    {"name": "Base (No HyDE/Rerank)", "hyde": False, "rerank": False},
    {"name": "Rerank Only",          "hyde": False, "rerank": True},
    {"name": "HyDE Only",            "hyde": True,  "rerank": False},
    {"name": "HyDE + Rerank",        "hyde": True,  "rerank": True},
]

collection = get_collection()
ctx = load_context("healer_mira")
results_summary = []

for config in configs:
    print(f"\n{'='*20} Evaluator: {config['name']} {'='*20}")
    print(f"{'Category':<25} | {'MAP':<8} | {'Coverage %':<12}")
    print("-" * 50)
    
    overall_ap = 0
    overall_queries = 0

    for category, data in TEST_DATA.items():
        cat_ap = 0
        cat_hits = 0
        cat_total_possible = 0
        
        queries = data["questions"]
        target_lists = data["answers"]

        for query, targets in zip(queries, target_lists):
            ctx.player_query = query
            retrieved = retrieve_lore(
                collection, 
                ctx, 
                use_hyde=config["hyde"], 
                use_crossencode=config["rerank"], 
                n_results=7
            )
            
            metrics = rank_retrieval(retrieved, targets)
            cat_ap += metrics["ap"]
            cat_hits += metrics["hits"]
            cat_total_possible += metrics["total"]
        
        cat_map = round(cat_ap / len(queries), 3)
        cat_coverage = round((cat_hits / cat_total_possible) * 100, 1)
        
        print(f"{category:<25} | {cat_map:<8} | {cat_coverage:<12}%")
        
        overall_ap += cat_ap
        overall_queries += len(queries)

    final_map = round(overall_ap / overall_queries, 3)
    print("-" * 50)
    print(f"{'TOTAL OVERALL MAP':<25} | {final_map:<8}")
    print("\n")
