from schemas import GameContext
import json
from config import NPCS_JSON

with open(NPCS_JSON, 'r', encoding='utf-8') as f:
    _raw = json.load(f)

CONTEXTS: dict[str, GameContext] = {
    key: GameContext(**val) for key, val in _raw.items()
}

def load_context(name):
    return CONTEXTS.get(name)
