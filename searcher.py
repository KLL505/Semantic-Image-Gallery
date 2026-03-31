import json
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel

class Searcher:
    def __init__(self, paths_file="paths.json"):
        self.paths_file = paths_file
        with open(self.paths_file, "r") as f:
            self.image_paths = json.load(f)

    #todo: implement the actual search logic using CLIP and FAISS. For now, it just returns all image paths regardless of the query.
    def search(self, query, top_k=3):
        print("Recieved query and top_k:", query, top_k)  
        return self.image_paths

    #todo: implement the actual logic to reload the index from disk. For now, it just re-reads the paths file and updates the image_paths list.
    def reload_index(self):
        with open(self.paths_file, "r") as f:
            self.image_paths = json.load(f)

        print("Search engine reloaded index")