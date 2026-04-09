import os
import gradio as gr
from searcher import Searcher
from indexer import Indexer

index_backend = Indexer()

#Only build the index on startup if the database file is missing
if not os.path.exists("embeddings.faiss"):
    print("Database not found. Building index for the first time...")
    index_backend.build_Index()
else:
    print("Database found. Skipping initial build.")

search_backend = Searcher()


def perform_search(query, top_k): 
    return search_backend.search(query, top_k)

def rebuild_index():
    index_backend.build_Index()
    search_backend.reload_index()
    return search_backend.image_paths

# -------------------------------------------------------------------
# Gradio Interface
# -------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Local Image Search") as app:
    gr.Markdown("# Local Semantic Image Search")
    
    with gr.Row():
        search_query = gr.Textbox(
            label="Search", 
            placeholder="Leave blank to see all images, or type a query...",
            scale=4 
        )
        top_k_slider = gr.Slider(
            minimum=1, maximum=50, value=3, step=1, 
            label="Max Results",
            scale=1 
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)
        rebuild_btn = gr.Button("Rebuild Index", variant="secondary", scale=1)
        
    results_gallery = gr.Gallery(
        label="Gallery", 
        show_label=False, 
        columns=5, 
        object_fit="scale-down", 
        height="100%",
        interactive=False
    )

    # Search
    search_btn.click(fn=perform_search, inputs=[search_query, top_k_slider], outputs=results_gallery)
    search_query.submit(fn=perform_search, inputs=[search_query, top_k_slider], outputs=results_gallery)
    
    # Rebuild
    rebuild_btn.click(fn=rebuild_index,inputs=[], outputs=results_gallery)

    # Initial load sends blank search to show all images
    app.load(fn=perform_search, inputs=[search_query, top_k_slider], outputs=results_gallery) 


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)
