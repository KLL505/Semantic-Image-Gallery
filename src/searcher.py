import json
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel

class Searcher:
    def __init__(self, device, model, processor, paths_file="./data/paths.json", index_file="./data/embeddings.faiss"):
        self.paths_file = paths_file # filename of list of image paths
        self.index_file = index_file # filename of embedding database
        self.image_paths = [] # list of image paths
        self.embedding_index = None # actual embedding database
        self.device = device
        self.model = model
        self.processor = processor

        # initial sync
        self.reload_index()

    '''
    Searches for top_k images using query
    :query: str|Image
    :top_k: number
    :return: list of image paths
    '''
    def search(self, query, top_k=3, maximum_images=50, offset=0):
        print(f"query={query} | top_k={top_k}")

        # base case: if the search query is empty, show everything
        if query is None or (isinstance(query, str) and not query.strip()):
            return self.image_paths[offset:offset + maximum_images]
    
        # base case: if the embedding index is empty, show nothing
        if self.embedding_index is None:
            return []

        # since we are using the model and not training it, we don't need to track gradients to save memory
        with torch.no_grad():
            # call the processor
            # move the inputs to device, safer than self.processor().to(self.device)
            # extract features from the encoder
            if isinstance(query, str):
                # query with text
                inputs = self.processor(text=[query], return_tensors="pt", padding=True)
                inputs = { k: v.to(self.device) for k, v in inputs.items() }
                features = self.model.get_text_features(**inputs)
            else:
                # query with image
                inputs = self.processor(images=query, return_tensors="pt")
                inputs = { k: v.to(self.device) for k, v in inputs.items() }
                features = self.model.get_image_features(**inputs)

            # normalize the vector to prepare for cosine similarity
            features /= features.norm(dim=-1, keepdim=True)

            # text_features is a tensor on self.device
            # we need to move it back to CPU because FAISS works on CPU
            # FAISS also works with numpy array and not tensor, so we need to convert back to numpy array
            query_vector = features.cpu().numpy().astype('float32')

        # search nearest k neighbors
        # dim(indices) = (num_queries, top_k). in this case, num_queries = 1 since we only have 1 query string
        distances, indices = self.embedding_index.search(query_vector, top_k)

        # map the resulting indices to actual file paths from the metadata
        # need to remove indices that are outside of valid range
        results = [self.image_paths[idx] for idx in indices[0] if idx >= 0 and idx < len(self.image_paths)]

        return results

    '''
    Loads image paths and embedding database into memory
    '''
    def reload_index(self):
        # loads the image paths from the paths
        try:
            with open(self.paths_file, "r") as f:
                self.image_paths = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # handle missing file or corrupted file
            self.image_paths = []
            print(f"Warning: {self.paths_file} missing or corrupt.")
        
        # loads the embedding database
        try:
            self.embedding_index = faiss.read_index(self.index_file)
            print(f"Index reloaded: {self.embedding_index.ntotal} images available")
        except Exception:
            print(f"Warning: {self.index_file} could not be loaded.")
            # handle error while loading file by adding a dummy index to prevent search crashes
            self.embedding_index = faiss.IndexFlatIP(512)

        print("Search engine reloaded index")