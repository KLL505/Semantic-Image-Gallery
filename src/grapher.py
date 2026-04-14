import torch
import pandas as pd

class Grapher:
    def __init__(self, device, model, processor, searcher):
        self.device = device
        self.model = model
        self.processor = processor
        self.searcher = searcher

    def generate_plot_data(self, x_axis_text, y_axis_text):
        # Base case: missing inputs or empty database
        if not x_axis_text or not y_axis_text:
            return pd.DataFrame() 
        if self.searcher.embedding_index is None or self.searcher.embedding_index.ntotal == 0:
            return pd.DataFrame()

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

        # Extract all raw image embeddings directly from the FAISS index
        ntotal = self.searcher.embedding_index.ntotal
        try:
            image_vectors = self.searcher.embedding_index.reconstruct_n(0, ntotal)
        except Exception as e:
            print(f"Error extracting vectors from FAISS: {e}")
            return pd.DataFrame()

        # Calculate Cosine Similarity scores for all images against the two text vectors
        x_scores = (image_vectors @ vec_x.T).flatten()
        y_scores = (image_vectors @ vec_y.T).flatten()

        # Build a Pandas DataFrame formatting the data
        df = pd.DataFrame({
            "Image": self.searcher.image_paths[:ntotal],
            x_axis_text: x_scores,
            y_axis_text: y_scores
        })
        
        return df