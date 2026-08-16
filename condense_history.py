import anthropic
from config import ANTHROPIC_API_KEY, MODEL_CONDENSE

MODEL_NAME = MODEL_CONDENSE

def condense_history(uncondensed):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    system_prompt = (
        "Condense the following conversation history into short, succinct bulletpoints. "
        "Cover what was asked and what the responses were. Respond only with these bulletpoints."
    )
    
    user_content = f"The conversation to condense: {str(uncondensed)}"

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_content}
        ]
    )
    
    summary_text = response.content[0].text.strip()

    return summary_text
