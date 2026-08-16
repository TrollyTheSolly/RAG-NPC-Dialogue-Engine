from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GameContext:
    npc_name: str
    player_query: str
    scene_description: str
    npc_personality: str = ""
    conversation_history: list = field(default_factory=list)
    summary_history: str = ""
    dialogue_examples: Optional[list[dict]] = field(default_factory=list)
    recent_lore: list[str] = field(default_factory=list)


@dataclass
class GeneratedDialogue:
    npc_name: str
    dialogue: str
    retrieved_context: list[str]
    prompt_used: str