import time
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import os

class SystemEvaluator:
    def __init__(self, searcher):
        self.searcher = searcher

    @staticmethod
    def calculate_map(retrieved_paths, relevant_paths):
       # We strip paths here too just to be safe
        retrieved_filenames = [os.path.basename(p) for p in retrieved_paths]
        relevant_filenames = [os.path.basename(p) for p in relevant_paths]

        if not relevant_filenames:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(retrieved_filenames):
            if p in relevant_filenames:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(relevant_filenames)

    def run_benchmark(self, ground_truth, top_k=20):
        # Run this to prevent warm-up/cold start delay which will result in the first query having a large latency
        self.warm_up()

        metrics = []
        latencies = []

        for item in ground_truth:
            query = item['query']
            relevant = item['relevant_paths'] 

            # Capture start and end time when we run a query
            start_time = time.perf_counter()
            results = self.searcher.search(query, top_k=top_k) # Assuming this returns full paths
            end_time = time.perf_counter()

            # Latency (ms)
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # NORMALIZE: Convert full paths to filenames only
            norm_results = [os.path.basename(p) for p in results]

            # Recall@5: Normalized comparison
            recall_at_5 = 1 if any(p in relevant for p in norm_results[:5]) else 0

            # nDCG
            y_true = [1 if p in relevant else 0 for p in norm_results]
            y_score = [10 - i for i in range(len(norm_results))]
            ndcg = ndcg_score([y_true], [y_score]) if len(norm_results) > 0 else 0

            # mAP
            map_val = self.calculate_map(norm_results, relevant)

            metrics.append({
                "Query": query,
                "Latency(ms)": round(latency_ms, 2),
                "Recall@5": recall_at_5,
                "mAP ": round(map_val, 3),
                "nDCG": round(ndcg, 3),
                "Ground Truth Count": len(relevant),
            })

        return pd.DataFrame(metrics), latencies

    def warm_up(self, dummy_query="dogs playing"):
        print("Warming up the system...")
        # Run a dummy search to trigger loading/caching
        self.searcher.search(dummy_query, top_k=5)
        print("System warmed up.")