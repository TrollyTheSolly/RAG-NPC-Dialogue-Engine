import csv
import io
import re
import json
 
import anthropic
 
from generate import generate_dialogue, get_collection
from contexts import CONTEXTS
from config import ANTHROPIC_API_KEY, MODEL_JUDGE, EVAL_LLM_JSON, WORLD_NAME
 
JUDGE_MODEL_NAME = MODEL_JUDGE
 
client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )

with open(EVAL_LLM_JSON, 'r', encoding='utf-8') as f:
    TEST_CASES = [tuple(case) for case in json.load(f)]
 
SYSTEM_PROMPT = (
    f"You are an expert narrative designer for '{WORLD_NAME}'. "
    "Your output must be a single line of CSV text. "
    "Do not include headers, explanations, or markdown code blocks."
    "Example output: 8; 7; Pass"
)
 
 
def build_user_prompt(npc_name, personality, lore, player_query, dialogue):
    return f"""
Evaluate this dialogue based on Persona (1-10), Lore (1-10), and Formatting (Pass/Fail).
If the dialogue contains hallucinated lore not present in the LORE section, heavily penalize it.
If the dialogue contains any "stage direction" or text that would be inappropriate to say aloud, fail its formatting.

SCORING RUBRIC:
Persona scoring (1-10):
  1-3: Wrong tone, contradicts character personality entirely
  4-6: Recognizable character but noticeable inconsistencies
  7-8: Good fit, minor deviations
  9-10: Indistinguishable from the source material

Lore scoring (1-10):
  1-3: Contradicts established lore or heavily hallucinates
  4-6: Mostly accurate but contains invented details
  7-8: Accurate, minor omissions
  9-10: Perfectly grounded in provided lore
 
[CHARACTER: {npc_name}]
Personality: {personality}
 
[LORE]
{lore}
 
[PLAYER QUERY]
{player_query}
 
[AI RESPONSE]
"{dialogue}"
 
Return exactly one line in this CSV format:
persona_score; lore_score; formatting_pass
"""
 
 
def evaluate_response(npc_name, personality, lore, player_query, dialogue):
    user_prompt = build_user_prompt(npc_name, personality, lore, player_query, dialogue)
 
    try:
        response = client.messages.create(
            model=JUDGE_MODEL_NAME,
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
 
        raw_text = response.content[0].text.strip()
        clean_text = re.sub(r"```(?:csv)?|```", "", raw_text).strip()
 
        reader = csv.reader(io.StringIO(clean_text), delimiter=';')
        row = next(reader)
 
        return {
            "persona_score":   row[0].strip(),
            "lore_score":      row[1].strip(),
            "formatting_pass": row[2].strip(),
        }
 
    except Exception as e:
        return {
            "error": f"Parsing failed: {str(e)}",
            "raw":   raw_text if "raw_text" in locals() else "No response",
        }
 
 
def run_evaluation_suite():
    collection = get_collection()
    results = []
 
    for npc_key, query in TEST_CASES:
        print(f"Testing {npc_key}...")
 
        ctx = CONTEXTS[npc_key]
        ctx.player_query = query
 
        rag_result = generate_dialogue(ctx, collection=collection)
        lore_text = "\n".join(rag_result.retrieved_context)
        evaluation = evaluate_response(
            ctx.npc_name, ctx.npc_personality, lore_text, query, rag_result.dialogue
        )
 
        results.append({
            "npc":            npc_key,
            "query":          query,
            "rag_dialogue":   rag_result.dialogue,
            "rag_scores":     evaluation,
            "lore_retrieved": rag_result.retrieved_context,
        })
 
    save_results_to_csv(results)
    print("\nEvaluation complete. Results saved to 'evaluation_results.csv'.")
 
 
def save_results_to_csv(results):
    with open("evaluation_results.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "NPC", "Query", "Mode", "Dialogue", "Persona Score", "Lore Score", "Formatting"
        ])
 
        for r in results:
            scores = r["rag_scores"]
 
            if "error" in scores:
                writer.writerow([
                    r["npc"], r["query"], "Rag", "ERROR",
                    0, 0, "Fail", scores["error"],
                ])
            else:
                writer.writerow([
                    r["npc"], r["query"], "Rag", r["rag_dialogue"],
                    scores.get("persona_score"),
                    scores.get("lore_score"),
                    scores.get("formatting_pass"),
                ])
 
 
if __name__ == "__main__":
    run_evaluation_suite()
