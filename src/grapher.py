import torch
import pandas as pd

class Grapher:
    def __init__(self, device, model, processor, searcher):
        self.device = device
        self.model = model
        self.processor = processor
        self.searcher = searcher

    def generate_plot_data(self, x_axis_text, y_axis_text, offset=0, max_graph_images=500):
        # Base case: missing inputs or empty database
        if not x_axis_text or not y_axis_text:
            return pd.DataFrame() 
        if self.searcher.embedding_index is None or self.searcher.embedding_index.ntotal == 0:
            return pd.DataFrame()
        offset = int(offset)

        with torch.no_grad():
            # Process the text for X and Y axes
            inputs_x = self.processor(text=[x_axis_text], return_tensors="pt", padding=True)
            inputs_y = self.processor(text=[y_axis_text], return_tensors="pt", padding=True)
            
            # Move to hardware
            inputs_x = {k: v.to(self.device) for k, v in inputs_x.items()}
            inputs_y = {k: v.to(self.device) for k, v in inputs_y.items()}
            
            # Generate text embeddings
            vec_x = self.model.get_text_features(**inputs_x)
            vec_y = self.model.get_text_features(**inputs_y)
            
            # Normalize vectors to prepare for cosine similarity
            vec_x /= vec_x.norm(dim=-1, keepdim=True)
            vec_y /= vec_y.norm(dim=-1, keepdim=True)
            
            # Convert back to numpy arrays
            vec_x = vec_x.cpu().numpy().astype('float32')
            vec_y = vec_y.cpu().numpy().astype('float32')

        # Calculate bounding indices for the 500 image limit and offset
        ntotal = self.searcher.embedding_index.ntotal
        start_idx = min(max(0, offset), ntotal)
        end_idx = min(start_idx + max_graph_images, ntotal)
        
        # If the offset pushes past available images, return empty
        if start_idx == end_idx:
            return pd.DataFrame()

        try:
            # Extract raw image embeddings directly from FAISS and slice the batch
            all_vectors = self.searcher.embedding_index.reconstruct_n(0, ntotal)
            image_vectors = all_vectors[start_idx:end_idx]
        except Exception as e:
            print(f"Error extracting vectors from FAISS: {e}")
            return pd.DataFrame()

        # Calculate Cosine Similarity scores for the SLICE against the text vectors
        x_scores = (image_vectors @ vec_x.T).flatten()
        y_scores = (image_vectors @ vec_y.T).flatten()

        # Build a Pandas DataFrame matching the sliced paths to the scores
        df = pd.DataFrame({
            "Image": self.searcher.image_paths[start_idx:end_idx],
            x_axis_text: x_scores,
            y_axis_text: y_scores
        })
        
        return df