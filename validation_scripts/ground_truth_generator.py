import pandas as pd
import json
import os

def compile_verified_gt(csv_path="validation_pool.csv", output_json="ground_truth.json"):
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Filter only for the ones you marked as 1
    verified = df[df['is_relevant'] == 1]
    
    ground_truth = []
    for query in verified['query'].unique():
        # Get list of files marked as 1 for this query
        relevant_paths = verified[verified['query'] == query]['filename'].tolist()
        ground_truth.append({
            "query": query,
            "relevant_paths": relevant_paths
        })
    
    with open(output_json, "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"Compiled Ground Truth with {len(verified)} verified images to {output_json}.")

# Run this once
# compile_verified_gt()