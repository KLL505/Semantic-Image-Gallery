# Semantic Image Search

**Final Project for CS 6384 Computer Vision**: A search engine that uses **CLIP (Contrastive Language-Image Pretraining)** and **FAISS** to perform semantic searches on local image datasets.

```bash
# Install Dependencies (if you use Conda, you don't need to run this)
pip install -r requirements.txt

# Run Application
python app.py
```

### Dataset Used

Flickr Dataset containing 31k Images from Kaggle: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
Validation Dataset which is a subset of 3k random images from the FLickr Dataset: https://drive.google.com/file/d/1xi3XttFL1OhjRHQo-BqFyUI8VqoF6NsM/view?usp=sharing

## Environment Setup with Conda

Prereq: Check `https://www.anaconda.com/docs/getting-started/miniconda/install/mac-gui-install` on how to install Conda.

I installed `Miniconda` from `https://www.anaconda.com/download/success?reg=skipped`

```bash
# 1. Create the environment (First time only)
conda env create -f environment.yml

# 2. Activate
conda activate image-search-app

# 3. Update/Create environment (Run this if the .yml file changes later)
conda env update --file environment.yml  --prune

# 4. Run Application
python app.py

# 4.1 Run Application on Watch Mode
gradio app.py

# 5. Open Application (server name and port are hard-coded in app.py)
http://127.0.0.1:7860

# 6. Deactivate
conda deactivate
```

## Troubleshooting for Conda

- **VS Code "Import could not be resolved":** Press `Cmd+Shift+P` -> `Python: Select Interpreter` -> Choose `image-search-app`.
- If you cannot find `image-search-app`, then in your terminal or command line:
  - First, make sure your conda environment is active.
  - Run `conda info --envs`.
  - Look for `image-search-app` and copy the full path next to it. It usually looks like `/Users/yourname/miniconda3/envs/image-search-app`.
  - Press `Cmd+Shift+P` -> `Python: Select Interpreter`
  - Paste the path you copied, but add `/bin/python` to the end of it -> hit `Enter`. For example, mine on a Macbook looks like `/opt/miniconda3/envs/image-search-app/bin/python`.
  - You can check if the interpreter has been applied using `which python`.

## Validation Scripts

Scripts used to generate validation and ground truth data files that are used when evaluating.    
Steps:
- Validation generator: generated validation_pool.csv with images marked as irrelevant "0" by sampling some queries using clip
- The images were manually reviewed and marked as "1" if they were relevant to the query
- ground truth generator: generated ground_truth.json using the manually reviewed validation_pool.csv
- The ground_truth.json was used for system evaluation
