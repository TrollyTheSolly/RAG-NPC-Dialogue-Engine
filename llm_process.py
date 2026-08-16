import json
import os
import time
import logging
from typing import List, Dict
import google.generativeai as genai
from google.api_core import exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from config import GEMINI_API_KEY, MODEL_HYDE, KNOWLEDGE_BASE_JSON, CURATED_FACTS_JSON, WORLD_NAME

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = MODEL_HYDE
model = genai.GenerativeModel(MODEL_NAME)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5

# This file was written with the assistance of an LLM
# Due to the cost and time associated with running these prompts, I needed help ensuring a robust structure.

@retry(
    retry=retry_if_exception_type((exceptions.InternalServerError, exceptions.ServiceUnavailable, exceptions.ResourceExhausted)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def distill_facts_robust(entity_name: str, category: str, dialogue_list: List[str]) -> List[str]:
    if not dialogue_list:
        return []

    combined_dialogue = "\n".join([f"- {line}" for line in dialogue_list])
    prompt = f"""
    You are an expert archivist for the world of {WORLD_NAME}. 
    Below is a list of dialogue snippets regarding the {category[:-1]} named "{entity_name}".
    
    TASK:
    Extract a list of concise, objective facts or lore details about "{entity_name}" based ONLY on the provided dialogue. 
    
    OUTPUT FORMAT:
    Return a JSON object with a single key "facts" containing a list of strings.
    """

    response = model.generate_content(
        [prompt, f"DIALOGUE:\n{combined_dialogue}"],
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        result = json.loads(response.text)
        return result.get("facts", [])
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse JSON for {entity_name}: {e}")
        return []

def load_progress(output_file: str) -> Dict:
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Output file error")
    return {"People": {}, "Places": {}, "Items": {}}

def process_knowledge_base(input_file: str, output_file: str):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Source file {input_file} not found.")
        return

    curated_data = load_progress(output_file)
    consecutive_errors = 0

    for category, entities in source_data.items():
        if category not in curated_data:
            curated_data[category] = {}

        logger.info(f"--- Processing Category: {category} ---")
        
        for entity_name, dialogue in entities.items():
            if entity_name in curated_data[category] and curated_data[category][entity_name]:
                logger.info(f"Skipping {entity_name} (already processed)")
                continue

            try:
                logger.info(f"Distilling: {entity_name}")
                facts = distill_facts_robust(entity_name, category, dialogue)
                
                curated_data[category][entity_name] = facts
                consecutive_errors = 0 
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(curated_data, f, indent=4)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Critical error on {entity_name}: {e}")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical("Too many consecutive errors. Circuit breaker triggered. Stopping.")
                    return

    logger.info(f"Processing complete. Data saved to {output_file}")

if __name__ == "__main__":
    process_knowledge_base(KNOWLEDGE_BASE_JSON, CURATED_FACTS_JSON)
