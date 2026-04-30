import os
import gradio as gr
from src.searcher import Searcher
from src.indexer import Indexer
from src.grapher import Grapher
from src.graph_component import generate_html_plot
from src.settings import Settings
from src.evaluator import SystemEvaluator
import json
import numpy as np

settings = Settings()

def format_gallery_results(paths, all_paths, offset=0):
    global total
    results = []
    for path in paths:
        filename = os.path.basename(path)
        caption = f"{filename}"
        results.append((path, caption))
    total = min(settings.max_index_images ,len(all_paths))
    end_index = min(offset + settings.max_results_empty, total)
    info_txt = f"<div style='text-align: right; font-size: 0.9em; color: #6b7280; padding-top: 4px; padding-right: 8px;'>Model: {settings.current_model_id} | Total Images: {total} | [{offset}-{end_index}]</div>"
    return results, info_txt

def perform_search(text_query, image_query, top_k, page=1):
    top_k = int(top_k)

    if page is not None:
        offset = (int(page) - 1) * settings.max_results_empty
    else:
        offset = 0
    if image_query is not None:
        paths = search_backend.search(image_query, top_k)
    else:
        paths = search_backend.search(text_query, top_k, settings.max_results_empty, offset)
    
    return format_gallery_results(paths, search_backend.image_paths, offset)

def rebuild_index():
    global index_backend, search_backend, graph_backend
    device, model, processor = settings.initialize_model(settings.current_model_id)
    
    index_backend = Indexer(device, model, processor, settings.img_dir)
    search_backend = Searcher(device, model, processor)
    graph_backend = Grapher(device, model, processor, search_backend)

    index_backend.build_Index(settings.batch_size, settings.max_index_images)
    search_backend.reload_index()
    
    paths = search_backend.image_paths[:settings.max_results_empty]
    
    return format_gallery_results(paths, search_backend.image_paths)

def change_settings_and_rebuild(new_model_id, img_dir, empty_max, batch_size, max_index, max_graph):
    settings.save_settings(new_model_id, img_dir, empty_max, batch_size, max_index, max_graph)
    return rebuild_index()


def generate_graph(x_text, y_text, offset):
    df = graph_backend.generate_plot_data(x_text, y_text, offset, settings.max_graph_images)
    if df.empty:
        return "<div style='text-align:center; padding:50px;'>Not enough data to plot or offset out of bounds.</div>"

    html_plot = generate_html_plot(df, x_text, y_text)
    return html_plot

def run_evaluation():
    with open("validation_scripts/ground_truth.json", "r") as f:
        gt = json.load(f)
    evaluator = SystemEvaluator(search_backend)
    df, latencies = evaluator.run_benchmark(gt)
    print('df',df)
    avg_map = df['mAP '].mean()
    avg_ndcg = df['nDCG'].mean()
    avg_latency = df['Latency(ms)'].mean()
    latency_std = np.std(latencies)
    summary = (
        f"### System Performance Summary\n"
        f"- **Model:** {settings.current_model_id}\n"
        f"- **Mean Latency:** {avg_latency:.2f} ms (±{latency_std:.2f}ms)\n"
        f"- **Mean mAP:** {avg_map:.3f}\n"
        f"- **Mean nDCG:** {avg_ndcg:.3f}\n"
    )
    
    model_name = settings.current_model_id.replace("/", "_")
    if df is None or df.empty:
        return gr.DownloadButton(visible=False)
    file_path = f"./data/evaluation_{model_name}.csv"
    df.to_csv(file_path, index=False)

    return gr.Dataframe(value=df, visible=True), gr.DownloadButton(value=file_path, visible=True), summary

# -------------------------------------------------------------------
# Gradio Interface
# -------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Semaintic Image Gallery", fill_height=True) as app:
    with gr.Row():
        gr.Markdown("# Semantic Image Gallery")
        info_txt = gr.HTML()

# -------------------------------------------------------------------
# Gallery
# -------------------------------------------------------------------    
    with gr.Tabs():
        with gr.Tab("Gallery"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    search_query = gr.Textbox(
                        label="Search by Text", 
                        placeholder="Leave blank to see all images, or type a query...",
                        max_lines=1
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
                        page_offset = gr.Number(
                            label="Page", 
                            value=1, 
                            minimum=1, 
                            precision=0
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
                        height="85vh", 
                        interactive=False
                    )
                    
            # Search
            page_offset.change(fn=perform_search, inputs=[search_query, image_query, top_k_slider, page_offset], outputs=[results_gallery, info_txt])
            search_btn.click(fn=perform_search, inputs=[search_query, image_query, top_k_slider, page_offset], outputs=[results_gallery, info_txt])
            search_query.submit(fn=perform_search, inputs=[search_query, image_query, top_k_slider, page_offset], outputs=[results_gallery, info_txt])
            
            # Rebuild
            rebuild_btn.click(fn=rebuild_index, inputs=[], outputs=[results_gallery, info_txt])

            # Initial load sends blank search to show all images
            app.load(fn=perform_search, inputs=[search_query, image_query, top_k_slider], outputs=[results_gallery, info_txt]) 

# -------------------------------------------------------------------
# Graph View
# -------------------------------------------------------------------
        with gr.Tab("Graph View"):
            with gr.Row():
                x_axis_input = gr.Textbox(label="X-Axis",placeholder="Nature", scale=2)
                y_axis_input = gr.Textbox(label="Y-Axis",placeholder="Industrial", scale=2)
                offset_input = gr.Number(label="Index Offset", value=0, precision=0, scale=1)
                plot_btn = gr.Button("Map Latent Space", variant="primary", scale=1)
            
            with gr.Row():
                latent_plot = gr.HTML(label="Latent Space")
                
            plot_btn.click(fn=generate_graph, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            x_axis_input.submit(fn=generate_graph, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            y_axis_input.submit(fn=generate_graph, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)
            offset_input.submit(fn=generate_graph, inputs=[x_axis_input, y_axis_input, offset_input], outputs=latent_plot)

# -------------------------------------------------------------------
# Model Benchmark
# -------------------------------------------------------------------
        with gr.Tab("Model Benchmark"):

            with gr.Row():
                eval_btn = gr.Button("Run Model Benchmark", variant="primary")
                export_btn = gr.DownloadButton("Export Results to CSV", visible=False)
                metrics_summary = gr.Markdown("Click run to generate metrics.")
            eval_table = gr.Dataframe(visible=False, max_height=1000)

            eval_btn.click(
                fn=run_evaluation, 
                outputs=[eval_table, export_btn, metrics_summary]
            )

            with gr.Accordion("Understanding Evaluation Metrics", open=True):
                gr.Markdown("""
                ### Evaluation Metrics Descriptions
                * **Recall@K**: Measures the ability of the system to find *at least one* relevant item in the top *K* results. It indicates the "coverage" of your search.
                * **mAP (Mean Average Precision)**: The average of precision scores calculated at the rank of each relevant item. It penalizes systems that push relevant items further down the list.
                * **nDCG (Normalized Discounted Cumulative Gain)**: A ranking quality metric. It accounts for the *position* of relevant items, assigning more "gain" to items retrieved at the very top (rank 1) compared to those at the bottom (rank 10).
                * **Latency(ms)**: The round-trip time for a query, measuring system responsiveness.
                """)
                gr.Markdown("""
                ### Metric Interpretation Guide
                | Metric | Score = 1 (Best) | Score = 0 (Worst) | What "High" vs "Low" Means |
                | :--- | :--- | :--- | :--- |
                | **Recall@K** | Found at least one match. | Found zero matches. | **Higher** is better (Coverage). |
                | **mAP** | Perfect ranking/precision. | No relevant results found. | **Higher** is better (Retrieval Quality). |
                | **nDCG** | Perfect ordering of results. | No relevant results found. | **Higher** is better (Ranking Sensitivity). |
                | **Latency** | N/A | N/A | **Lower** is better (Speed). |
                """)

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
        with gr.Tab("Settings"):
                    gr.Markdown("### Application Settings")
                    with gr.Row():
                        with gr.Column(scale=2):
                            model_input = gr.Textbox(
                                label="Hugging Face Model ID", 
                                value=settings.current_model_id, 
                                info="Embedding model used for indexing and searching. Warning: Changing the model will completely overwrite your current FAISS database."
                            )
                            img_dir_input = gr.Textbox(
                                label="Image Directoy Path", 
                                value=settings.img_dir, 
                                info="Image directory path application will load from"
                            )
                            empty_max_input = gr.Number(
                                label="Max Results on Empty Search",
                                value=settings.max_results_empty,
                                precision=0,
                                info="How many images to display per page when the search query is blank."
                            )
                            batch_size_input = gr.Number(
                                label="Index Batch Size",
                                value=settings.batch_size,
                                precision=0,
                                info="Number of images to process at once while indexing. Lower this if your GPU runs out of memory."
                            )
                            max_index_input = gr.Number(
                                label="Max Images to Index",
                                value=settings.max_index_images,
                                precision=0,
                                info="Limit the total number of images read from the directory."
                            )
                            max_graph_input = gr.Number(
                                label="Max Images to Graph",
                                value=settings.max_graph_images,
                                precision=0,
                                info="Limit the total number of images top graph."
                            )
                        with gr.Column(scale=1):
                            apply_btn = gr.Button("Apply & Rebuild Index", variant="primary")
                            status_text = gr.Markdown("") 
                            
                    apply_btn.click(
                        fn=lambda: "Applying settings and rebuilding database... Please wait.", 
                        outputs=[status_text]
                    ).then(
                        fn=change_settings_and_rebuild, 
                        inputs=[model_input, img_dir_input, empty_max_input, batch_size_input, max_index_input, max_graph_input], 
                        outputs=[results_gallery, info_txt]
                    ).then(
                        fn=lambda: "Success! Settings saved and index rebuilt.", 
                        outputs=[status_text]
                    )

if __name__ == "__main__":
    shared_device, shared_model, shared_processor = settings.initialize_model()
    
    index_backend = Indexer(shared_device, shared_model, shared_processor, settings.img_dir)

    if not os.path.exists("./data/embeddings.faiss"):
        print("Database not found. Building index for the first time...")
        os.makedirs("./data", exist_ok=True)
        index_backend.build_Index()
    else:
        print("Database found. Skipping initial build.")

    search_backend = Searcher(shared_device, shared_model, shared_processor)
    graph_backend = Grapher(shared_device, shared_model, shared_processor, search_backend)

    app.launch(server_name="127.0.0.1", server_port=7860, share=True)