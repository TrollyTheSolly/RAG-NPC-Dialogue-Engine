import logging
import sys
from generate import get_collection, generate_dialogue
from contexts import CONTEXTS, load_context
from condense_history import condense_history
from config import INTERFACE_TITLE

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

CONDENSE_THRESHOLD = 400

def start_conversation():
    print(f"--- {INTERFACE_TITLE} ---")
    collection = get_collection()

    print("\nAvailable NPCs:")
    npc_keys = list(CONTEXTS.keys())
    for i, key in enumerate(npc_keys):
        print(f"[{i}] {key}")
    
    choice = input("\nSelect an NPC by number: ")
    if choice.lower() == 'exit':
        return

    try:
        selected_key = npc_keys[int(choice)]
        ctx = load_context(selected_key)
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    print(f"\n--- Entering Scene: {ctx.npc_name} ---")
    print(f"Location: {ctx.scene_description}")


    while True:
        if ctx.player_query != "":
            user_input = ctx.player_query
            print(f"Player: {user_input}")
        else:
            user_input = input("Player: ")

        ctx.player_query = user_input

        result = generate_dialogue(ctx, collection=collection, log_level=1, use_hyde=False)
        
        print(f"\n{ctx.npc_name}: {result.dialogue}\n")

        half = max(1, len(result.retrieved_context) // 2)
        ctx.recent_lore = result.retrieved_context[:half]

        total_chars = len(str(ctx.conversation_history))
        if (total_chars > CONDENSE_THRESHOLD):
            print("CONDENSING...")
            ctx.summary_history += condense_history(ctx.conversation_history)
            ctx.conversation_history = []

        ctx.conversation_history.append({
            "player": user_input,
            "npc": result.dialogue
        })


        ctx.player_query = "" 

if __name__ == "__main__":
    try:
        start_conversation()
    except KeyboardInterrupt:
        sys.exit(0)