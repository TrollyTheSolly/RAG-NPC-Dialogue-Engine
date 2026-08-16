import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
import chromadb
import logging
import google.generativeai as genai
from chromadb.utils import embedding_functions
from contexts import load_context
from schemas import GameContext, GeneratedDialogue
from sentence_transformers import CrossEncoder
import anthropic
from config import (
    GEMINI_API_KEY, ANTHROPIC_API_KEY, MODEL_HYDE, MODEL_DIALOGUE,
    VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, RERANKER_MODEL,
    WORLD_NAME, SETTING_NAME, get_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = MODEL_HYDE
DB_PATH = VECTOR_DB_PATH
N_RETRIEVAL_RESULTS = 10
RERANKER = CrossEncoder(RERANKER_MODEL)
TOP_K_CANDIDATES = 30


def get_collection():
    huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=get_device()
    )

    client = chromadb.PersistentClient(path=DB_PATH)

    return client.get_collection(name=COLLECTION_NAME, embedding_function=huggingface_ef)

def generate_hyde(query: str):
    model = genai.GenerativeModel(MODEL_NAME)
    hyde_prompt = f"""
    SYSTEM:
    You are an expert archivist for the world of {WORLD_NAME}. Your task is to assist in document retrieval by generating a hypothetical archival entry based on a user's inquiry.

    TASK:
    Write a list of 5-10 concise, objective facts or lore details that would likely appear in the archives regarding the user's query.
    Answer only the player's query, ignore unimportant text such as the name of who they are speaking to, or their situation.

    CONSTRAINTS:
    1. Use an objective tone.
    2. Avoid conversational language, flowery prose, or personal opinions.
    3. If the user asks a question, do not answer it directly; instead, provide the facts that would contain the answer.
    4. Begin your response with the first fact, include no additional dressing.

    USER QUERY: 
    {query}

    HYPOTHETICAL ARCHIVAL FACTS:
    """
    response = model.generate_content(hyde_prompt)

    return response.text.strip()


def retrieve_lore(collection, context: GameContext, n_results: int = N_RETRIEVAL_RESULTS, use_hyde = False, use_crossencode = True) -> list[str]:
    if use_hyde:
        hyde = generate_hyde(context.player_query)
        query_text = f"Represent this sentence for searching relevant passages: '{hyde}'"
    else:
        query_text = f"Represent this sentence for searching relevant passages: '{context.player_query}'"
    
    results = collection.query(
        query_texts=[query_text],
        n_results=TOP_K_CANDIDATES
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not use_crossencode:
        retrieved = [
            f"[{m.get('category', 'Unknown')}] {d}" 
            for d, m in zip(documents, metadatas)
        ]
        return retrieved[:n_results]

    

    pairs = [[context.player_query, doc] for doc in documents]
    
    scores = RERANKER.predict(pairs,show_progress_bar=False)

    scored_results = []
    for i in range(len(documents)):
        scored_results.append({
            "score": scores[i],
            "doc": documents[i],
            "category": metadatas[i].get("category", "Unknown")
        })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:n_results]

    retrieved = [f"[{res['category']}] {res['doc']}" for res in top_results]
    
    return retrieved


def build_prompt(context: GameContext, lore_chunks: list[str]) -> str:
    #Fromkeys to remove duplicates
    combined_lore = list(dict.fromkeys(lore_chunks + context.recent_lore))

    if not combined_lore:
        lore_block = "(No lore available. Do not invent or assume any world details.)"
    else:
        lore_block = "".join(f"  - {chunk}\n" for chunk in combined_lore)


    summary_block = ""

    if context.summary_history:
        summary_block = summary_block + context.summary_history

    history_block = ""

    if context.conversation_history:

        lines = []

        for turn in context.conversation_history:

            lines.append(f"  Player: {turn['player']}")

            lines.append(f"  {context.npc_name}: {turn['npc']}")

        history_block = "\n".join(lines) + "\n\n"



    persona_block = (

        f"{context.npc_personality}"

        if context.npc_personality

        else ""

    )



    examples_block = ""

    if context.dialogue_examples:

        lines = []

        for ex in context.dialogue_examples:

            lines.append(f"  Player: \"{ex['player']}\"")

            lines.append(f"  {context.npc_name}: \"{ex['npc']}\"")

        examples_block = "\n".join(lines)




    prompt = f"""
You are roleplaying as {context.npc_name}, a character living in {SETTING_NAME}.

CHARACTER PROFILE (canonical — takes priority over lore):
{persona_block}

BEHAVIOURAL RULES:
- Reveal emotion through subtext and deflection, not confession.
- Let contradictions sit unresolved. Avoid tidy emotional closure.
- If the player is vague or minimal, react to their tone, not their words.
- Never acknowledge you have said something before — simply move forward.
- Keep the conversation moving, if you have said something before do not repeat it.

WORLD LORE (inform your response; do not recite.  Avoid inventing lore or facts that are not confirmed here.  
If you are asked something not covered here, do not invent a response.):
{lore_block}

CURRENT SCENE:
{context.scene_description}

FEW-SHOT EXAMPLES:
{examples_block}

CONVERSATION SO FAR:
{summary_block}
{history_block}

OUTPUT FORMAT:
- 1-3 sentences of spoken dialogue only.
- Begin directly with the first word of dialogue.
- No stage directions, no name prefix, no fourth-wall breaks.
- If the player's input has no clear in-world referent, respond 
  with in-character confusion or deflection.

USER PROMPT:
{context.player_query}
"""

    print(prompt)
    
    return prompt


def generate_dialogue(context: GameContext, collection=None, log_level=0, use_hyde = False) -> GeneratedDialogue:
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )

    if collection is not None:
        lore_chunks = retrieve_lore(collection, context, use_hyde=use_hyde)
    else:
        lore_chunks = []

    prompt = build_prompt(context, lore_chunks)

    if log_level > 1:
        print()
        print('='*60)
        print(prompt)
        print('='*60)
        print()

    message = client.messages.create(
        model=MODEL_DIALOGUE,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    dialogue_text = message.content[0].text.strip()

    result = GeneratedDialogue(
        npc_name=context.npc_name,
        dialogue=dialogue_text,
        retrieved_context=lore_chunks,
        prompt_used=prompt,
    )

    if log_level >= 1:
        print("\n" + "="*60)
        if lore_chunks:
            print("\n[LORE RETRIEVED]")
            for chunk in lore_chunks:
                print(f"  • {chunk}")
        print("="*60 + "\n")

    return result

if __name__ == "__main__":
    collection = get_collection()

    ctx = load_context("innkeeper_bran")
    ctx.player_query = "Who runs the Silver Inn?"

    result = retrieve_lore(collection, ctx, use_hyde=False)

    print(result)