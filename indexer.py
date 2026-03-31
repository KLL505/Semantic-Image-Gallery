import os
import json
import torch
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class Indexer:

    def __init__(self,image_dir="./images", paths_file="paths.json"):
        self.image_dir = image_dir
        self.paths_file = paths_file

    #Builds paths file by scanning image directory for supported formats and saving their paths to a JSON file.
    def build_paths(self):
        image_paths = [
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        with open(self.paths_file, "w") as f:
            json.dump(image_paths, f)

        print(f"Found {len(image_paths)} images, Paths saved to {self.paths_file}.")

    #TODO: implement the actual logic to build the index using CLIP and FAISS. For now, it just builds the paths file.
    def build_Index(self):
        self.build_paths()


