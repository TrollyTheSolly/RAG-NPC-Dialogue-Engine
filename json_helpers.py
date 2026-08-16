import json

def clean_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cleaned = {}
    for category, entities in data.items():
        cleaned[category] = {k: v for k, v in entities.items() if v}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=4)
