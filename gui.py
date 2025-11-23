
import gradio as gr
import torch
from generate import Generator
import re
import random
import os

# --- Global variables for model caching ---
generator = None
last_config = {}

def process_wildcards(prompt):
    wildcard_matches = re.findall(r'__(\w+)__', prompt)
    if not wildcard_matches:
        return prompt

    for wildcard_name in wildcard_matches:
        wildcard_path = os.path.join("wildcards", f"{wildcard_name}.txt")
        if os.path.exists(wildcard_path):
            with open(wildcard_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                random_line = random.choice(lines)
                prompt = prompt.replace(f"__{wildcard_name}__", random_line, 1)
    return prompt

def generate_image(mode, prompt, negative_prompt, resolution, use_grpo, device, num_inference_steps, guidance_scale, num_images_per_prompt, batch_count, keep_in_memory):
    global generator, last_config

    processed_prompt = process_wildcards(prompt)
    print(f"Original prompt: {prompt}")
    print(f"Processed prompt: {processed_prompt}")

    current_config = {
        "mode": mode,
        "resolution": resolution,
        "device": device,
        "no_grpo": not use_grpo
    }

    if generator is None or current_config != last_config:
        if generator is not None:
            del generator
            torch.cuda.empty_cache()
        generator = Generator(**current_config)
        last_config = current_config

    all_image_paths = []
    # The batch_count from the UI can be a float if the user enters it, so we cast to int.
    for i in range(int(batch_count)):
        print(f"\n--- Batch {i+1} of {int(batch_count)} ---")
        try:
            # The numeric inputs from the UI are strings, so we cast them to the correct types.
            image_paths = generator.generate(processed_prompt, negative_prompt, int(num_inference_steps), float(guidance_scale), int(num_images_per_prompt), "output")
            all_image_paths.extend(image_paths)
        except Exception as e:
            print(f"Error during image generation: {e}")
            if not keep_in_memory:
                del generator
                generator = None
                last_config = {}
                torch.cuda.empty_cache()
            return None

    if not keep_in_memory:
        print("Clearing model from memory.")
        del generator
        generator = None
        last_config = {}
        torch.cuda.empty_cache()

    return all_image_paths

with gr.Blocks() as demo:
    gr.Markdown("# Nitro-E Image Generation")
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", placeholder="A majestic __color__ cat in a forest", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="low quality, blurry, watermark", lines=2)
            with gr.Row():
                num_inference_steps = gr.Textbox(value="20", label="Inference Steps")
                guidance_scale = gr.Textbox(value="4.5", label="Guidance Scale")
            with gr.Row():
                num_images_per_prompt = gr.Textbox(value="1", label="Batch Size (per run)")
                batch_count = gr.Textbox(value="1", label="Batch Count (runs)")
            with gr.Row():
                mode = gr.Dropdown(choices=['20_steps', '4_steps'], label="Mode", value='20_steps')
                resolution = gr.Dropdown(choices=[512, 1024], label="Resolution", value=512)
                use_grpo = gr.Checkbox(label="Use GRPO", value=True)
            with gr.Row():
                device = gr.Textbox(label="Device", value="cuda:0")
                keep_in_memory = gr.Checkbox(label="Keep model in memory", value=True)
            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery")

    generate_button.click(
        fn=generate_image, 
        inputs=[
            mode, 
            prompt, 
            negative_prompt, 
            resolution, 
            use_grpo, 
            device, 
            num_inference_steps, 
            guidance_scale, 
            num_images_per_prompt, 
            batch_count,
            keep_in_memory
        ],
        outputs=output_gallery,
    )

demo.launch()
