import os
import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel
from src.searcher import Searcher
from src.indexer import Indexer
from src.grapher import Grapher
from src.graph_component import generate_html_plot

#initializes backend classes with shared model and processor instances to save memory and load time. Also handles device setup for GPU/CPU/MPS.
def initialize_backend():
    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")

    # load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    processor.use_fast = False

    return device, model, processor

def perform_search(text_query, image_query, top_k):
    top_k = int(top_k)
    if image_query is not None:
        return search_backend.search(image_query, top_k)
    return search_backend.search(text_query, top_k)

def rebuild_index():
    index_backend.build_Index()
    search_backend.reload_index()
    return search_backend.image_paths

def update_latent_plot(x_text, y_text):
    df = graph_backend.generate_plot_data(x_text, y_text)
    if df.empty:
        return "<div style='text-align:center; padding:50px;'>Not enough data to plot.</div>"

    html_plot = generate_html_plot(df, x_text, y_text)
    return html_plot

# -------------------------------------------------------------------
# Gradio Interface
# -------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Local Image Search") as app:
    gr.Markdown("# Local Semantic Image Search")
    
    with gr.Tabs():
        with gr.Tab("Semantic Search"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    search_query = gr.Textbox(
                        label="Search by Text", 
                        placeholder="Leave blank to see all images, or type a query...",
                        lines=2,
                        max_lines=5
                    )
                    image_query = gr.Image(
                        label="Search by Image",
                        type="pil",
                        height=250,
                        sources=["upload"]
                    )
                    with gr.Group():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=3,
                            step=1, 
                            label="Max Results"
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    rebuild_btn = gr.Button("Rebuild Index", variant="secondary")
                with gr.Column(scale=3):
                    results_gallery = gr.Gallery(
                        label="Gallery", 
                        show_label=False, 
                        columns=4, 
                        rows=4,
                        object_fit="scale-down", 
                        height="600px",
                        interactive=False
                    )

            # Search
            search_btn.click(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=results_gallery)
            search_query.submit(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=results_gallery)
            
            # Rebuild
            rebuild_btn.click(fn=rebuild_index,inputs=[], outputs=results_gallery)

            # Initial load sends blank search to show all images
            app.load(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=results_gallery) 

        with gr.Tab("Latent Space Graph"):
            gr.Markdown("Map your images across two distinct semantic concepts.")
            with gr.Row():
                x_axis_input = gr.Textbox(label="X-Axis",placeholder="Nature", scale=2)
                y_axis_input = gr.Textbox(label="Y-Axis",placeholder="Industrial", scale=2)
                plot_btn = gr.Button("Map Latent Space", variant="primary", scale=1)
            
            with gr.Row():
                latent_plot = gr.HTML(label="Latent Space")
                
            plot_btn.click(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input], outputs=latent_plot)
            x_axis_input.submit(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input], outputs=latent_plot)
            y_axis_input.submit(fn=update_latent_plot, inputs=[x_axis_input, y_axis_input], outputs=latent_plot)


if __name__ == "__main__":
    index_backend = Indexer(*initialize_backend())

    #Only build the index on startup if the database file is missing
    if not os.path.exists("./data/embeddings.faiss"):
        print("Database not found. Building index for the first time...")
        os.makedirs("./data", exist_ok=True)
        index_backend.build_Index()
    else:
        print("Database found. Skipping initial build.")

    search_backend = Searcher(*initialize_backend())
    graph_backend = Grapher(*initialize_backend(), search_backend)

    app.launch(server_name="127.0.0.1", server_port=7860, share=True)
