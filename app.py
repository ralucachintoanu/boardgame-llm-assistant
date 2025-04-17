import gradio as gr
import json
from game_assist import build_prompt_with_rag, generate_response

with open("games.json", "r") as f:
    games = json.load(f)
games = {k: v["name"] for k, v in games.items()}


def clear_output():
    return ""

custom_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate"
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("## 🎲 Game Rules Assistant")

    with gr.Row():
        game_selector = gr.Dropdown(
            choices=[(label, key) for key, label in games.items()],
            label="Choose a game",
            value=None
        )
        top_k_selector = gr.Dropdown(
            choices=[3, 4, 5, 6, 7],
            label="Top K Context Results",
            value=5
        )
        model_selector = gr.Dropdown(
            choices=[
                ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                ("microsoft/phi-2 (better, but slower)", "microsoft/phi-2"),
            ],
            label="Choose a model",
            value="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )        

    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(
                lines=4,
                label="Prompt with context (RAG)"
            )
            generate_button = gr.Button("Generate Summary")

        output_box = gr.Textbox(
            lines=20,
            label="Answer",
            interactive=False
        )

    game_selector.change(
        fn=build_prompt_with_rag, 
        inputs=[game_selector, top_k_selector], 
        outputs=prompt_box)
    
    top_k_selector.change(
        fn=build_prompt_with_rag, 
        inputs=[game_selector, top_k_selector], 
        outputs=prompt_box)
    
    game_selector.change(
        fn=clear_output, 
        inputs=None, 
        outputs=output_box)
    
    model_selector.change(
        fn=clear_output, 
        inputs=None, 
        outputs=output_box)
    
    top_k_selector.change(
        fn=clear_output, 
        inputs=None, 
        outputs=output_box)

    generate_button.click(
        fn=generate_response, 
        inputs=[prompt_box, model_selector], 
        outputs=output_box)

demo.launch()
