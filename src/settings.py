import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel

class Settings:
    def __init__(self, settings_file="./data/settings.json"):
        self.settings_file = settings_file
        
        # Defaults
        self.default_model = "openai/clip-vit-base-patch32"
        self.default_img_dir = "./images"
        self.default_max_results_empty = 50
        self.default_batch_size = 32
        self.default_max_index_images = 3000
        self.default_max_graph_images = 500

        
        # Automatically load the saved settings
        settings = self.load_settings()
        self.current_model_id = settings.get("model_id", self.default_model)
        self.img_dir = settings.get("img_dir", self.default_img_dir)
        self.max_results_empty = settings.get("max_results_empty", self.default_max_results_empty)
        self.batch_size = settings.get("batch_size", self.default_batch_size)
        self.max_index_images = settings.get("max_index_images", self.default_max_index_images)
        self.max_graph_images = settings.get("max_graph_images", self.default_max_graph_images)


    def load_settings(self):
        # Start with a clean dictionary of defaults
        defaults = {
            "model_id": self.default_model,
            "img_dir": self.default_img_dir,
            "max_results_empty": self.default_max_results_empty,
            "batch_size": self.default_batch_size,
            "max_index_images": self.default_max_index_images,
            "max_graph_images": self.default_max_graph_images
        }
        
        # If the file exists, read it and overwrite any defaults with saved values
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    saved_settings = json.load(f)
                    defaults.update(saved_settings)
                    return defaults
            except Exception as e:
                print(f"Error loading settings: {e}")
                return defaults
        else:
            # If the file DOES NOT exist, create it automatically with the defaults
            print("Settings file not found. Creating a default settings.json...")
            self.save_settings(
                defaults["model_id"], 
                defaults["img_dir"],
                defaults["max_results_empty"], 
                defaults["batch_size"], 
                defaults["max_index_images"],
                defaults["max_graph_images"]
            )
            return defaults

    def save_settings(self, model_id, img_dir, max_results_empty, batch_size, max_index_images, max_graph_images):
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        
        # Structure the new JSON payload
        settings_dict = {
            "model_id": model_id,
            "img_dir": img_dir,
            "max_results_empty": int(max_results_empty),
            "batch_size": int(batch_size),
            "max_index_images": int(max_index_images),
            "max_graph_images": int(max_graph_images)

        }
        
        with open(self.settings_file, "w") as f:
            json.dump(settings_dict, f, indent=4)
            
        # Update the active class trackers
        self.current_model_id = model_id
        self.img_dir = img_dir
        self.max_results_empty = int(max_results_empty)
        self.batch_size = int(batch_size)
        self.max_index_images = int(max_index_images)
        self.max_graph_images = int(max_graph_images)
        

    def initialize_model(self, model_id=None):
        if model_id is None:
            model_id = self.current_model_id
            
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        print(f"Using {device} device for {model_id}")

        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval()
        processor.use_fast = False

        return device, model, processor