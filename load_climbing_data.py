import pandas as pd
import requests
from elasticsearch import Elasticsearch, helpers
from zipfile import ZipFile
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import time
from datetime import datetime
import tiktoken

MAX_TOKENS_PER_BATCH = 8191


def download_and_load_data():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    data_file = data_dir / "climbing_data.pkl.zip"
    
    if not data_file.exists():
        print("Downloading data file...")
        url = "https://github.com/OpenBeta/climbing-data/raw/main/curated_datasets/CuratedWithRatings_OpenBetaAug2020_RytherAnderson.pkl.zip"
        response = requests.get(url)
        
        data_file.write_bytes(response.content)
        print("Data file downloaded and saved locally.")
    else:
        print("Using cached data file.")
    
    with ZipFile(data_file) as zip_file:
        pkl_filename = zip_file.namelist()[0]
        with zip_file.open(pkl_filename) as pkl_file:
            df = pd.read_pickle(pkl_file)
    return df

def get_embeddings_for_batch(text: list[str], openai_client: OpenAI) -> list[float]:
    # NOTE: fails on empty strings
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        start_time = time.time()
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[encoding.encode(line) for line in text]
        )
        duration = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Embedding API call took {duration:.2f}s")
        return [elt.embedding for elt in response.data]
    except Exception as e:
        print(f"Error getting embedding for text: {text}")
        print(e)
        return []

_last_progress_time = None
def _print_progress(start_time, processed_count, total_routes):
    global _last_progress_time
    current_time = time.time()

    # Print progress every 10 seconds or every 100 routes
    if _last_progress_time is None or current_time - _last_progress_time > 10 or (processed_count % 500 == 0 and processed_count > 0):
        elapsed_time = current_time - start_time
        routes_per_second = processed_count / elapsed_time
        estimated_remaining = (total_routes - processed_count) / routes_per_second if routes_per_second > 0 else 0
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Processed {processed_count}/{total_routes} routes "
                f"({processed_count/total_routes*100:.1f}%) - "
                f"Rate: {routes_per_second:.1f} routes/s - "
                f"Est. remaining: {estimated_remaining/60:.1f} minutes")
        _last_progress_time = current_time

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def add_embeddings(documents: dict, openai_client: OpenAI):
    batch_start_idx = 0
    batch_end_idx = 0 # exclusive
    batch_token_size = 0
    start_time = time.time()
    document_ids = [document_id for document_id, document in documents.items() if document["description"] != ""]
    while batch_start_idx < len(document_ids):
        _print_progress(start_time, batch_start_idx, len(document_ids))
        if batch_end_idx < len(document_ids):
            next_doc = documents[document_ids[batch_end_idx]]
            next_doc_tokens = num_tokens_from_string(next_doc["description"])
            
            while next_doc_tokens >= MAX_TOKENS_PER_BATCH:
                # case where one doc alone has too many tokens -> need to truncate description
                next_doc["description"] = next_doc["description"][0:len(next_doc["description"]) // 2]
                next_doc_tokens = num_tokens_from_string(next_doc["description"])

            if batch_token_size + next_doc_tokens < MAX_TOKENS_PER_BATCH/2:
                # existing batch has room, keep adding docs to it
                batch_token_size += next_doc_tokens
                batch_end_idx += 1
                continue
        
        # process the batch
        print(f"Getting embeddings for batch of length {batch_end_idx - batch_start_idx}")
        embeddings = get_embeddings_for_batch([documents[doc_id]["description"] for doc_id in document_ids[batch_start_idx:batch_end_idx]], openai_client)
        for doc_id, embedding in zip(document_ids[batch_start_idx:batch_end_idx], embeddings):
            documents[doc_id]['description_vector'] = embedding
        batch_start_idx = batch_end_idx
        batch_token_size = next_doc_tokens
        batch_end_idx += 1

def transform_data(df):
    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def extract_coordinates(location_str):
        try:
            if pd.isna(location_str):
                return None
            lat, lon = map(float, location_str.split(','))
            return {"lat": lat, "lon": lon}
        except:
            return None

    documents = []
    total_routes = len(df)
    start_time = time.time()
    _last_progress_time = start_time
    
    print(f"Starting to transform {total_routes} routes...")
    
    for idx, row in df.iterrows():
        _print_progress(start_time=start_time, processed_count=idx, total_routes=total_routes)       
        
        description = "\n".join(row["description"] or [])
        doc = {
            "route_name": row["route_name"],
            "route_id": row["route_ID"],
            "sector_id": row["sector_ID"],
            "grade": row["YDS"] if pd.notna(row["YDS"]) else row["Vermin"],
            "sector_name": row["parent_sector"],
            "location": extract_coordinates(row["parent_loc"]),
            "style": row["type_string"],  # trad, sport, mixed, or boulder
            "description": description,
            "description_vector": None,
            "rating": float(np.mean([rating[1] for rating in row["corrected_users_ratings"]])) if isinstance(row["corrected_users_ratings"], (list, np.ndarray)) and len(row["corrected_users_ratings"]) > 0 else None
        }
        # Only include document if it has valid data
        if doc["route_name"] and doc["grade"] and doc["route_id"]:
            documents.append(doc)
    
    print("Adding embeddings for all routes...")
    
    add_embeddings({document['route_id']: document for document in documents}, openai_client)

    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time/60:.1f} minutes")
    print(f"Average processing time per route: {total_time/len(documents):.2f} seconds")
    
    return documents

def load_to_elasticsearch(documents):
    load_dotenv()
    
    es_url = os.getenv('ELASTICSEARCH_URL')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')
    index_name = os.getenv('ELASTICSEARCH_INDEX_NAME', 'openbeta')
    
    if es_api_key:
        es = Elasticsearch(
            es_url,
            api_key=es_api_key
        )
    else:
        raise Exception('Missing required env var: ELASTICSEARCH_API_KEY')
    
    mappings = {
        "properties": {
            "route_name": {"type": "text"},
            "route_id": {"type": "keyword"},
            "sector_id": {"type": "keyword"},
            "grade": {"type": "keyword"},
            "sector_name": {"type": "keyword"},
            "location": {"type": "geo_point"},
            "style": {"type": "keyword"},
            "description": {"type": "text"},
            "description_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
            "rating": {"type": "float"}
        }
    }
    
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    
    es.indices.create(index=index_name, mappings=mappings)
    
    actions = [
        {
            "_index": index_name,
            "_source": doc
        }
        for doc in documents
    ]
    
    helpers.bulk(es, actions)

def main():
    print("Downloading and loading data...")
    df = download_and_load_data()
    
    print("Transforming data...")
    documents = transform_data(df)
    
    print("Loading data to Elasticsearch...")
    load_to_elasticsearch(documents)
    print("Done!")

if __name__ == "__main__":
    main()
