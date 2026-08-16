import spacy
from collections import defaultdict
import json
import csv
from json_helpers import clean_json
from config import DIALOGUE_CSV, KNOWLEDGE_BASE_JSON

gpu_activated = spacy.prefer_gpu()

if gpu_activated:
    print("Success! Using GPU:", spacy.util.get_installed_models())
else:
    print("Fail: Running on CPU. Check your CUDA installation.")

nlp = spacy.load("en_core_web_trf")

def generate_entity_dictionary(csv_path, output_json=KNOWLEDGE_BASE_JSON):
    documents = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('content', '').strip()
                if text:
                    documents.append(text)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return {}

    entity_dict = {
        "People": defaultdict(list),
        "Places": defaultdict(list),
        "Items": defaultdict(list),
        "Factions_and_Groups": defaultdict(list),
        "History_and_Timeline": defaultdict(list),
        "Regulations": defaultdict(list),
    }

    total_length = len(documents)
    print(f"Processing {total_length} lines of dialogue...")

    for doc in nlp.pipe(documents, batch_size=64, disable=["lemmatizer"]):
        for ent in doc.ents:
            category = None
            if ent.label_ == "PERSON":
                category = "People"
            elif ent.label_ in ["GPE", "FAC", "LOC"]:
                category = "Places"
            elif ent.label_ in ["PRODUCT", "WORK_OF_ART"]:
                category = "Items"
            elif ent.label_ in ["NORP", "ORG"]:
                category = "Factions_and_Groups"
            elif ent.label_ in ["EVENT", "DATE"]:
                category = "History_and_Timeline"
            elif ent.label_ == "LAW":
                category = "Regulations"
            
            if category:
                entity_name = ent.text.strip().title()
                try:
                    sentence_context = ent.sent.text.strip()
                except ValueError:
                    sentence_context = doc.text

                if sentence_context not in entity_dict[category][entity_name]:
                    entity_dict[category][entity_name].append(sentence_context)

    final_output = {}
    for category, items in entity_dict.items():
        final_output[category] = dict(items)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Dictionary complete! Saved to {output_json}")
    return final_output


if __name__ == '__main__':
    entity_data = generate_entity_dictionary(DIALOGUE_CSV)
    clean_json(KNOWLEDGE_BASE_JSON)
