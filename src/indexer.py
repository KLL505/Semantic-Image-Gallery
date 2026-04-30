import os
import json
import torch
import numpy as np
import faiss
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Indexer:

    def __init__(self, device, model, processor, image_dir="./images", paths_file="./data/paths.json", index_file="./data/embeddings.faiss"):
        self.image_dir = image_dir
        self.paths_file = paths_file
        self.index_file = index_file
        
        self.device = device
        self.model = model
        self.processor = processor

    #Builds paths file by scanning image directory for supported formats and saving their paths to a JSON file.
    def build_paths(self):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        image_paths = [
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        with open(self.paths_file, "w") as f:
            json.dump(image_paths, f)

        print(f"Found {len(image_paths)} images. Paths saved to {self.paths_file}.")
        return image_paths

    def build_Index(self, batch_size=32, num_images=3000):
        all_paths = self.build_paths()
        
        #If num_images is provided, we limit to that number (for testing)
        image_paths = all_paths[:num_images] if num_images else all_paths
        
        if not image_paths:
            print("No images found to index. Skipping index building.")
            return

        print(f"Indexing a subset of {len(image_paths)} images in batches of {batch_size}...")
        embeddings_list = []
        
        start_time = time.perf_counter()
        for i in range(0, len(image_paths), batch_size):
            #Slicing: takes images from i to i + 32
            batch_paths = image_paths[i : i + batch_size] 
            batch_images = []
            
            for path in batch_paths:
                try:
                    #Open and process the image
                    image = Image.open(path).convert("RGB")
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    
            if not batch_images:
                continue
                
            try:
                #Processes the actual batch of 32
                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    batch_features = self.model.get_image_features(**inputs)
                
                batch_features = batch_features / batch_features.norm(p=2, dim=-1, keepdim=True)
                embeddings_list.extend(batch_features.cpu().numpy().astype('float32'))
                
                print(f"Progress: {min(i + batch_size, len(image_paths))} / {len(image_paths)}")
                
            except Exception as e:
                print(f"Batch error at {i}: {e}")

        #Convert to a single 2D numpy matrix of float32
        embeddings_matrix = np.array(embeddings_list).astype('float32')
        
        #FAISS index
        faiss.normalize_L2(embeddings_matrix)
        dimension = embeddings_matrix.shape[1]
        
        #Inner Product: equivalent to Cosine Similarity for CLIP
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)
        faiss.write_index(index, self.index_file)
        end_time = time.perf_counter()
        print(f"Successfully built FAISS index with {index.ntotal} vectors in {end_time - start_time} seconds.")
        print(f"Database saved to {self.index_file}")