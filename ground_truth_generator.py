import pandas as pd
import json
import os

synonyms = {
    "dog": ["dog", "puppy", "canine", "hound"],
    "play": ["play", "playing", "running", "fun", "frolic"],
    "child": ["child", "kid", "baby", "boy", "girl", "toddler"],
    "outside": ["outside", "park", "grass", "field", "beach", "sun"],
    "city": ["city", "urban", "street", "building", "town", "downtown"],
    "night": ["night", "dark", "evening", "midnight", "lights"],
    "music": ["guitar", "music", "playing", "band", "concert", "instrument"],
    "water": ["water", "ocean", "sea", "lake", "pool", "river"],
    "sports": ["sports", "running", "jump", "ball", "race", "athletic"],
    "sun": ["sun", "sunny", "bright", "daylight", "summer"]
}

def is_relevant(caption, terms, synonyms):
    caption = str(caption).lower()
    # Check if ANY of the synonyms for your terms appear in the caption
    for term in terms:
        # If the term has synonyms, check for any of them
        search_terms = synonyms.get(term, [term])
        if not any(s in caption for s in search_terms):
            return False # Must have at least one synonym for every term
    return True

def build_ground_truth(csv_path, image_dir, output_path="ground_truth.json"):
    local_files = set(os.listdir(image_dir))
    df = pd.read_csv(csv_path, sep="|")
    df.columns = df.columns.str.strip()
    
    queries = [
        {"query": "dogs playing", "terms": ["dog", "play"], "min_images": 5},
        {"query": "people outdoors", "terms": ["child", "outside"], "min_images": 5},
        {"query": "urban life", "terms": ["city"], "min_images": 5},
        {"query": "musical performance", "terms": ["music"], "min_images": 3},
        {"query": "near water", "terms": ["water"], "min_images": 5},
        {"query": "night settings", "terms": ["night"], "min_images": 3},
        {"query": "athletic activity", "terms": ["sports"], "min_images": 3},
        {"query": "sunny weather", "terms": ["sun"], "min_images": 5},
        {"query": "childhood activities", "terms": ["child", "play"], "min_images": 5},
        {"query": "casual walking", "terms": ["run"], "min_images": 5}
    ]
    
    ground_truth = []
    
    for item in queries:
        mask = df['comment'].apply(lambda x: is_relevant(x, item['terms'], synonyms))
        matches = df[mask & df['image_name'].isin(local_files)]['image_name'].unique().tolist()
        
        # Only add to ground truth if we found enough images
        if len(matches) >= item['min_images']:
            ground_truth.append({
                "query": item['query'],
                "relevant_paths": matches[:10] # Cap at 10
            })
            
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"Generated {len(ground_truth)} queries.")

def initialize_ground_truth(csv_path, image_dir):
    data_dir = "data"
    output_path = os.path.join(data_dir, "ground_truth.json")
    
    # Check if file exists to avoid unnecessary re-computation
    if os.path.exists(output_path):
        print(f"Ground truth found at {output_path}. Skipping generation.")
        return
    
    # If we are here, file doesn't exist. Create the folder and run the generator.
    print("Ground truth not found. Generating...")
    os.makedirs(data_dir, exist_ok=True)
    
    build_ground_truth(csv_path, image_dir, output_path)
    print(f"Generated and saved to {output_path}")