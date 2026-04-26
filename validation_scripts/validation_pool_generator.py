import pandas as pd
import os
from src.searcher import Searcher
from src.settings import Settings

settings = Settings()
shared_device, shared_model, shared_processor = settings.initialize_model()
search_backend = Searcher(shared_device, shared_model, shared_processor)

queries = [
    {"query": "dogs playing", "terms": ["dog", "play"], "min_images": 5},
    {"query": "people outdoors", "terms": ["child", "outside"], "min_images": 5},
    {"query": "urban life", "terms": ["city"], "min_images": 5},
    {"query": "musical performance", "terms": ["music"], "min_images": 3},
    {"query": "near water", "terms": ["water"], "min_images": 5},
    {"query": "night settings", "terms": ["night"], "min_images": 3},
    {"query": "athletic activity", "terms": ["sports"], "min_images": 3},
    {"query": "sunny weather", "terms": ["sun"], "min_images": 5},
    {"query": "childhood activities", "terms": ["child", "play"], "min_images": 5},
    {"query": "casual walking", "terms": ["run"], "min_images": 5}
]

def export_validation_candidates(searcher, queries, output_csv="validation_pool.csv", top_n=20):
    candidates = []
    print("Generating candidate pool from validation_images...")
    
    for item in queries:
        query = item['query']
        # Ensure your searcher is pointing to validation_images
        results = searcher.search(query, top_k=top_n)
        
        for res in results:
            candidates.append({
                "query": query,
                "filename": '../validation_images/' + os.path.basename(res),
                "is_relevant": 0  # Default to 0
            })
    
    df = pd.DataFrame(candidates)
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(candidates)} candidates to {output_csv}. Open this in Excel/Sheets to label.")

# Run this once
export_validation_candidates(search_backend, queries)