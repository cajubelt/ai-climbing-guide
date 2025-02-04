from constants import ELASTICSEARCH_INDEX_NAME
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
from core.embedding import add_embeddings


def download_and_load_data():
    data_dir = Path(os.getenv("DATA_DIR", "data"))
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

def extract_coordinates(location):
    try:
        if pd.isna(location).any():  
            print("Location contains NaN values:", location)
            return None
        if not hasattr(location, '__len__') or len(location) != 2:
            print("Expected location to be a sequence of length 2, got", location)
            return None
        lon, lat = float(location[0]), float(location[1])  
        if not (np.isfinite(lon) and np.isfinite(lat)):
            print("Expected coordinates to be finite numbers, got", lon, lat)
            return None
        return {"lat": lat, "lon": lon}
    except Exception as e:
        print("Exception extracting coordinates:", e)
        return None

def transform_data(df):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    documents = []
    total_routes = len(df)
    start_time = time.time()
    _last_progress_time = start_time
    
    print(f"Starting to transform {total_routes} routes...")
    
    for _, row in df.iterrows():
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
    
    es_url = os.getenv('ELASTICSEARCH_NODE_URL')
    es_api_key = os.getenv('ELASTICSEARCH_API_KEY')
    index_name = ELASTICSEARCH_INDEX_NAME
    
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
            "sector_name": {"type": "text"},
            "description": {"type": "text"},
            "location": {"type": "geo_point"},
            "rating": {"type": "float"},
            "style": {"type": "keyword"},
            "grade": {"type": "keyword"},
            "route_id": {"type": "keyword"},
            "sector_id": {"type": "keyword"},
            "description_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
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
    load_dotenv()

    print("Downloading and loading data...")
    df = download_and_load_data()
    
    print("Transforming data...")
    documents = transform_data(df)
    
    print("Loading data to Elasticsearch...")
    load_to_elasticsearch(documents)
    print("Done!")

if __name__ == "__main__":
    main()
