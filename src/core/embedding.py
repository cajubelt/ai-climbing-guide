import time
import tiktoken
from openai import OpenAI
import json
from datetime import datetime
from pathlib import Path
import os

MAX_TOKENS_PER_BATCH = 8191


def get_embeddings_for_batch(text: list[str], openai_client: OpenAI) -> list[list[float]]:
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
    if _last_progress_time is None or current_time - _last_progress_time > 10:
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
    print('Loading embedding cache')
    embedding_cache = load_embedding_cache()
    print(f'Loaded embedding cache, found {len(embedding_cache)} cached embeddings')
    batch_token_size = 0
    start_time = time.time()
    batch = []
    def process_batch():
        print(f"Getting embeddings for batch of length {len(batch)}")
        embeddings = get_embeddings_for_batch([doc["description"] for doc in batch], openai_client)
        for doc, embedding in zip(batch, embeddings):
            doc['description_vector'] = embedding
            embedding_cache[doc["description"]] = embedding
        save_embedding_cache(embedding_cache)
    for idx, next_doc in enumerate(documents.values()):
        _print_progress(start_time, idx, len(documents))
        description = next_doc["description"]
        next_doc_tokens = num_tokens_from_string(description)
        if description == "":
            # openai api requires non-emptystring description to get an embedding
            continue
        while next_doc_tokens >= MAX_TOKENS_PER_BATCH:
            # case where one doc alone has too many tokens -> need to truncate description
            description = description[0:len(description) // 2]
            next_doc_tokens = num_tokens_from_string(description)
        next_doc['description'] = description  # need to update in case we truncated

        if batch_token_size + next_doc_tokens >= MAX_TOKENS_PER_BATCH:
            process_batch()
            batch = []
            batch_token_size = 0
        
        if description in embedding_cache:
            next_doc['description_vector'] = embedding_cache[description]
        else:
            batch_token_size += next_doc_tokens
            batch.append(next_doc)
    
    process_batch() if len(batch) > 0 else None  # process the last batch if nonempty


def _get_embedding_cache_file_path():
    data_dir = Path(os.getenv("DATA_DIR", "../../data"))
    data_dir.mkdir(exist_ok=True)
    return data_dir / "embedding_cache.json"


def load_embedding_cache() -> dict[str, list[float]]:
    embedding_cache_file_path = _get_embedding_cache_file_path()
    if embedding_cache_file_path.exists():
        return json.load(open(embedding_cache_file_path))
    else:
        return {}


def save_embedding_cache(embedding_cache: dict[str, list[float]]):
    embedding_cache_file_path = _get_embedding_cache_file_path()
    json.dump(embedding_cache, open(embedding_cache_file_path, "w"))
