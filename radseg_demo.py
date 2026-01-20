import sys
import os
import logging
import base64
from io import BytesIO
import colorsys
from functools import partial

import gradio as gr
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm

# Add current directory to sys.path to find radseg module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from radseg.radseg import RADSegEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global state to keep track of prompts and colors (similar to inspiration app)
prompt_list = []
color_list = []

def apply_colormap(image: np.ndarray, cmap_name='viridis') -> np.ndarray:
    """Apply a colormap to a grayscale image and return an RGB uint8 image."""
    if image.dtype != np.float16 and image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0, 1)
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(image)[:, :, :3]  # Drop alpha channel
    return (colored * 255).astype(np.uint8)

def numpy_to_base64(img_array):
    """Convert a NumPy array image to base64 string."""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def make_grid_output(images, labels):
    html = """
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
    """
    for img_array, label in zip(images, labels):
        img_str = numpy_to_base64(img_array)
        html += f"""
        <div style='text-align: center;'>
            <div style='font-weight: bold; margin-bottom: 5px;'>{label}</div>
            <img src='data:image/png;base64,{img_str}' style='width: 100%; height: auto; border: 1px solid #ccc;' />
        </div>
        """
    html += "</div>"
    return html

def generate_distinct_color(index):
    """Generate visually distinct colors using HSV color space."""
    hue = (index * 0.61803398875) % 1  # golden ratio
    r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def add_prompt(prompts):
    for prompt in prompts.split("\n"):
        prompt = prompt.strip()
        if not prompt or prompt in prompt_list:
            continue
        color = generate_distinct_color(len(prompt_list))
        prompt_list.append(prompt)
        color_list.append(color)

    colored_prompts = [
        f"<span style='background-color:{color}; color:#FFFFFF; padding: 2px 5px; border-radius: 3px; margin: 2px; display: inline-block;'>{p}</span>" 
        for p, color in zip(prompt_list, color_list)
    ]
    return gr.update(value=""), gr.update(value=" ".join(colored_prompts))

def clear_prompts():
    prompt_list.clear()
    color_list.clear()
    return gr.update(value=""), gr.update(value="")

def on_page_load():
    prompt_list.clear()
    color_list.clear()
    return gr.update(value="")

# Load model lazily
_encoder_cache = {}

def get_encoder(model_version, lang_model, scra_scaling, scga_scaling, slide_crop, slide_stride):
    cache_key = (model_version, lang_model, scra_scaling, scga_scaling, slide_crop, slide_stride)
    if cache_key not in _encoder_cache:
        logger.info(f"Loading encoder: {model_version} with {lang_model}")
        try:
            # We don't pass classes yet as we will handle that in inference
            # or by manually encoding prompts.
            # To avoid the "Must pass list of classes when predict is True" error,
            # we'll use predict=False for the base encoder and manage predictions ourselves.
            enc = RADSegEncoder(
                model_version=model_version,
                lang_model=lang_model,
                scra_scaling=scra_scaling,
                scga_scaling=scga_scaling,
                slide_crop=slide_crop,
                slide_stride=slide_stride,
                sam_refinement=False,
                predict=False, 
                device=device
            )
            _encoder_cache[cache_key] = enc
        except Exception as e:
            logger.error(f"Error loading encoder: {e}")
            raise gr.Error(f"Failed to load encoder: {e}")
    return _encoder_cache[cache_key]

@torch.inference_mode()
def process_all(input_image, scra_scaling, scga_scaling, use_sliding_window, window_size, window_stride, softmax, resolution):
    use_templates = True
    model_version = "c-radio_v3-b"
    lang_model = "siglip2"
    N = len(prompt_list)
    if N == 0:
        raise gr.Error("You must add some prompts", duration=5)
    
    yield "Initializing encoder..."
    slide_crop = window_size if use_sliding_window else 0
    slide_stride = window_stride if use_sliding_window else 224
    encoder = get_encoder(model_version, lang_model, scra_scaling, scga_scaling, slide_crop, slide_stride)

    yield "Processing image..."
    # Convert numpy to torch
    orig_h, orig_w = input_image.shape[:2]
    
    # Resize to requested resolution (maintaining aspect ratio)
    if max(orig_h, orig_w) > resolution:
        scale = resolution / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        # Ensure divisible by 16 for VIT/RADIO if needed, though slide crop might handle it
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16
        input_image_resized = np.array(Image.fromarray(input_image).resize((new_w, new_h), resample=Image.BILINEAR))
    else:
        input_image_resized = input_image

    image_size = input_image_resized.shape[:2]
    tensor_image = torch.from_numpy(input_image_resized).permute(2, 0, 1).to(device).float() / 255.0
    
    # Run RADSeg processing
    # Note: RADSegEncoder handle sliding window if slide_crop is set.
    # We can temporarily override resolution if needed but let's use the encoder's defaults.
    
    # Manual similarity calculation to match inspiration app's heatmap per prompt
    yield "Computing features..."
    feat_map = encoder.encode_image_to_feat_map(tensor_image.unsqueeze(0), orig_img_size=image_size)
    
    yield "Aligning features..."
    aligned_feats = encoder.align_spatial_features_with_language(feat_map, onehot=False)
    
    yield "Encoding prompts..."
    if use_templates:
        prompt_embeds = encoder.encode_labels(prompt_list)
    else:
        prompt_embeds = encoder.encode_prompts(prompt_list)
        
    yield "Computing similarity..."
    B, C, H, W = aligned_feats.shape
    aligned_feats_flat = aligned_feats.permute(0, 2, 3, 1).reshape(-1, C)
    
    # Compute cosine similarity
    # similarity: [N_prompts, H*W]
    vec1 = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
    vec2 = aligned_feats_flat / aligned_feats_flat.norm(dim=-1, keepdim=True)
    sim = (vec1 @ vec2.t()) # [N, H*W]
    
    if softmax:
        sim = torch.softmax(100 * sim, dim=0)
    else:
        sim /= torch.max(sim) # For visualization
        
    sim = sim.reshape(N, H, W)
    
    # Resize heatmaps to original resolution for display
    sim_resized = F.interpolate(sim.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
    
    yield "Generating outputs..."
    heatmaps = [apply_colormap(sim_resized[i].cpu().numpy()) for i in range(N)]
    yield make_grid_output(heatmaps, prompt_list)

def main():
    with gr.Blocks(title="RADSeg 2D Demo") as demo:
        gr.Markdown(
            """
            # RADSeg: Zero-Shot Open-Vocabulary Segmentation
            ### [Paper](https://arxiv.org/abs/2511.19704) | [Project Page](https://radseg-ovss.github.io/)
            Test RADSeg's zero-shot open-vocabulary segmentation capabilities on any image using text prompts !
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="numpy")

                with gr.Accordion("Model & Inference Settings", open=False):
                    with gr.Group():
                        gr.Markdown("### 1. Model Scaling")
                        scra_scaling = gr.Slider(1.0, 20.0, 10.0, step=1.0, label="SCRA Scaling", info="Self-Correlating Recursive Attention")
                        scga_scaling = gr.Slider(1.0, 20.0, 10.0, step=1.0, label="SCGA Scaling", info="Self-Correlating Global Aggregation")

                    with gr.Group():
                        gr.Markdown("### 2. Sliding Window")
                        use_sliding_window = gr.Checkbox(label="Enable Sliding Window", value=True)
                        with gr.Row():
                            window_size = gr.Slider(224, 1024, 336, step=16, label="Window Size")
                            window_stride = gr.Slider(112, 512, 224, step=16, label="Window Stride")

                    with gr.Row():
                        softmax = gr.Checkbox(label="Use softmax", value=True)
                        res_slider = gr.Slider(224, 1024, 512, step=32, label="Resolution (Max Side)")

                with gr.Group():
                    gr.Markdown("### 3. Text Prompts")
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt", 
                            placeholder="Type a class (e.g. 'car') and press enter",
                            scale=4,
                            show_label=False
                        )
                        add_button = gr.Button("+", scale=1)

                    prompt_display = gr.HTML()

                    with gr.Row():
                        clear_button = gr.Button("Clear Prompts")
                        run_button = gr.Button("Run Segmentation", variant="primary")

            with gr.Column(scale=2):
                output_html = gr.HTML(label="Segmentation Heatmaps")

        # Event handlers
        prompt.submit(add_prompt, inputs=prompt, outputs=[prompt, prompt_display])
        add_button.click(add_prompt, inputs=prompt, outputs=[prompt, prompt_display])
        clear_button.click(clear_prompts, outputs=[prompt, prompt_display])

        gr.Examples(
            examples=[
                ["assets/example1.jpg", 10, 10, True, 336, 224, True, "Pothole\nRoad\nSky\nCar\nWater", 768],
                ["assets/example2.jpg", 10, 10, True, 336, 224, True, "Person\nShoes\nGrey Jacket\nRed overalls\nRoad\nCrosswalk\nCar", 768],
                ["assets/example3.jpg", 10, 10, True, 336, 224, True, "Paved ground\nFlood lights\nRed Container\nBuilding\nTanker\nSky\nClouds\nTreeline", 1024]
            ],
            inputs=[input_image, scra_scaling, scga_scaling, use_sliding_window, window_size, window_stride, softmax, prompt, res_slider],
        )

        run_button.click(
            fn=process_all,
            inputs=[
                input_image,
                scra_scaling, scga_scaling,
                use_sliding_window, window_size, window_stride,
                softmax, res_slider
            ],
            outputs=output_html
        )
        
        demo.load(on_page_load, outputs=prompt_display)

    demo.queue().launch(share=True)

if __name__ == "__main__":
    main()

