import chromadb
import json
from chromadb.utils import embedding_functions
from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, CURATED_FACTS_JSON, get_device

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL,
                    device=get_device()
                )
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=huggingface_ef)

json_file_path = CURATED_FACTS_JSON
BATCH_SIZE = 500 

def upload_json_in_batches():
    documents = []
    ids = []
    metadatas = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    global_count = 0
    
    # Iterate through Categories (People, Items, Places)
    for category, sub_content in data.items():
        # Iterate through Sub-keys (Example, etc.)
        for sub_key, entries in sub_content.items():
            # Iterate through the actual list of strings
            for entry in entries:
                contextual_document = f"({sub_key}): {entry}"
                documents.append(contextual_document)
                ids.append(f"id_{global_count}")
                
                metadatas.append({
                    "category": category,
                    "sub_category": sub_key
                })
                
                global_count += 1

                if len(documents) >= BATCH_SIZE:
                    collection.add(
                        documents=documents,
                        ids=ids,
                        metadatas=metadatas
                    )
                    print(f"Added batch up to entry {global_count}...")
                    documents, ids, metadatas = [], [], []

    if documents:
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Finished! Total records processed: {global_count}")

upload_json_in_batches()
