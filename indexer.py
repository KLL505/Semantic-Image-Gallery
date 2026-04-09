import os
import json
import torch
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class Indexer:

    def __init__(self,image_dir="./images", paths_file="paths.json", index_file="embeddings.faiss"):
        self.image_dir = image_dir
        self.paths_file = paths_file
        self.index_file = index_file

        #Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Builds paths file by scanning image directory for supported formats and saving their paths to a JSON file.
    def build_paths(self):
        #Check for directory first
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        image_paths = [
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        with open(self.paths_file, "w") as f:
            json.dump(image_paths, f)

        print(f"Found {len(image_paths)} images, Paths saved to {self.paths_file}.")
        return image_paths

    #TODO: implement the actual logic to build the index using CLIP and FAISS. For now, it just builds the paths file.
    def build_Index(self):
        image_paths = self.build_paths()
        
        if not image_paths:
            print("No images found to index. Skipping index building.")
            return
        
        #Load standard CLIP model and processor here
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print("Generating embeddings with CLIP... this might take a moment.")
        embeddings_list = []
        
        for path in image_paths:
            try:
                #Open and process the image
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                
                #Get image features (embeddings) without tracking gradients
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                
                #Normalize the embeddings (Cosine Similarity in FAISS)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                #Extract 1D numpy array
                embedding = image_features.cpu().numpy()[0]
                embeddings_list.append(embedding)
                
            except Exception as e:
                print(f"Error processing image {path}: {e}")
        
        if not embeddings_list:
            print("Failed to generate any embeddings.")
            return

        #Convert to a single 2D numpy matrix of float32
        embeddings_matrix = np.array(embeddings_list).astype('float32')
        
        #FAISS index
        faiss.normalize_L2(embeddings_matrix)
        dimension = embeddings_matrix.shape[1]
        
        #Inner Product: equivalent to Cosine Similarity for CLIP
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)
        faiss.write_index(index, self.index_file)
        
        print(f"Successfully built FAISS index with {index.ntotal} vectors.")
        print(f"Database saved to {self.index_file}")