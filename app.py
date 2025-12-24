import spaces
import json
import math
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Iterable
import re
import time
from threading import Thread
from io import BytesIO
import uuid
import tempfile

import gradio as gr
import numpy as np
import torch
from PIL import Image
import supervision as sv


from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Glm4vForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# --- Theme and CSS Definition ---

# Define the SteelBlue color palette
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",  # SteelBlue base color
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)


class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

# Instantiate the new theme
steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""


# --- Constants and Model Setup ---
MAX_INPUT_TOKEN_LENGTH = 4096
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- System Information ---")
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", device)
print("--------------------------")


# --- Model Loading ---

# Load Camel-Doc-OCR-062825
print("Loading Camel-Doc-OCR-062825...")
MODEL_ID_M = "prithivMLmods/Camel-Doc-OCR-062825"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()
print("Camel-Doc-OCR-062825 loaded.")

# GLM-4.1V-9B-Thinking
print("Loading GLM-4.1V-9B-Thinking")
MODEL_ID_T = "zai-org/GLM-4.1V-9B-Thinking"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True)
model_t = Glm4vForConditionalGeneration.from_pretrained(
    MODEL_ID_T,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()
print("GLM-4.1V-9B-Thinking loaded.")

# Load moondream3
print("Loading moondream3-preview...")
MODEL_ID_MD3 = "moondream/moondream3-preview"
model_md3 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_MD3,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda"},
)
model_md3.compile()
print("moondream3-preview loaded and compiled.")


# --- Moondream3 Utility Functions ---

def create_annotated_image(image, detection_result, object_name="Object"):
    if not isinstance(detection_result, dict) or "objects" not in detection_result:
        return image
    
    original_width, original_height = image.size
    annotated_image = np.array(image.convert("RGB"))
  
    bboxes = []
    labels = []
    
    for i, obj in enumerate(detection_result["objects"]):
        x_min = int(obj["x_min"] * original_width)
        y_min = int(obj["y_min"] * original_height)
        x_max = int(obj["x_max"] * original_width)
        y_max = int(obj["y_max"] * original_height)
        
        x_min = max(0, min(x_min, original_width))
        y_min = max(0, min(y_min, original_height))
        x_max = max(0, min(x_max, original_width))
        y_max = max(0, min(y_max, original_height))
        
        if x_max > x_min and y_max > y_min:
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(f"{object_name} {i+1}")
    
    if not bboxes:
        return image
        
    detections = sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        class_id=np.arange(len(bboxes))
    )
    
    bounding_box_annotator = sv.BoxAnnotator(
        thickness=3,
        color_lookup=sv.ColorLookup.INDEX
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=0.6,
        color_lookup=sv.ColorLookup.INDEX
    )
    
    annotated_image = bounding_box_annotator.annotate(
        scene=annotated_image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )
    
    return Image.fromarray(annotated_image)

def create_point_annotated_image(image, point_result):
    if not isinstance(point_result, dict) or "points" not in point_result:
        return image
    
    original_width, original_height = image.size
    annotated_image = np.array(image.convert("RGB"))
    
    points = []
    for point in point_result["points"]:
        x = int(point["x"] * original_width)
        y = int(point["y"] * original_height)
        points.append([x, y])
    
    if points:
        points_array = np.array(points).reshape(1, -1, 2)
        key_points = sv.KeyPoints(xy=points_array)
        vertex_annotator = sv.VertexAnnotator(radius=8, color=sv.Color.RED)
        annotated_image = vertex_annotator.annotate(
            scene=annotated_image, key_points=key_points
        )
    
    return Image.fromarray(annotated_image)

@spaces.GPU()
def detect_objects_md3(image, prompt, task_type, max_objects):
    STANDARD_SIZE = (1024, 1024)
    if image is None:
        raise gr.Error("Please upload an image.")
    image.thumbnail(STANDARD_SIZE)
    
    t0 = time.perf_counter()
    
    if task_type == "Object Detection":
        settings = {"max_objects": max_objects} if max_objects > 0 else {}
        result = model_md3.detect(image, prompt, settings=settings)
        annotated_image = create_annotated_image(image, result, prompt)
    elif task_type == "Point Detection":
        result = model_md3.point(image, prompt)
        annotated_image = create_point_annotated_image(image, result)
    elif task_type == "Caption":
        result = model_md3.caption(image, length="normal")
        annotated_image = image  
    else:  
        result = model_md3.query(image=image, question=prompt, reasoning=True)
        annotated_image = image  
          
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    
    if isinstance(result, dict):
        if "objects" in result:
          output_text = f"Found {len(result['objects'])} objects:\n"
          for i, obj in enumerate(result['objects'], 1):
              output_text += f"\n{i}. Bounding box: ({obj['x_min']:.3f}, {obj['y_min']:.3f}, {obj['x_max']:.3f}, {obj['y_max']:.3f})"
        elif "points" in result:
            output_text = f"Found {len(result['points'])} points:\n"
            for i, point in enumerate(result['points'], 1):
                output_text += f"\n{i}. Point: ({point['x']:.3f}, {point['y']:.3f})"
        elif "caption" in result:
            output_text = result['caption']
        elif "answer" in result:
            output_text = f"Reasoning: {result.get('reasoning', 'N/A')}\n\nAnswer: {result['answer']}"
        else:
            output_text = json.dumps(result, indent=2)
    else:
        output_text = str(result)
    
    timing_text = f"Inference time: {elapsed_ms:.0f} ms"
    
    return annotated_image, output_text, timing_text


# --- Core Application Logic (for other models) ---
@spaces.GPU
def process_document_stream(
    model_name: str, 
    image: Image.Image, 
    prompt_input: str, 
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float
):
    """
    Main generator function for models other than Moondream3.
    """
    if image is None:
        yield "Please upload an image."
        return
    if not prompt_input or not prompt_input.strip():
        yield "Please enter a prompt."
        return

    # Select processor and model based on dropdown choice
    if model_name == "Camel-Doc-OCR-062825 (OCR)":
        processor, model = processor_m, model_m
    elif model_name == "GLM-4.1V-9B (Thinking)":
        processor, model = processor_t, model_t
    else:
        yield "Invalid model selected."
        return
            
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_input}]}]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_full], images=[image], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH).to(device)
    
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs, 
        "streamer": streamer, 
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": True if temperature > 0 else False
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        # Clean up potential model-specific tokens
        buffer = buffer.replace("<|im_end|>", "").replace("</s>", "")
        time.sleep(0.01)
        yield buffer

def create_gradio_interface():
    """Builds and returns the Gradio web interface."""

    with gr.Blocks() as demo:
        gr.Markdown("# **Multimodal VLM v1.0**", elem_id="main-title")
        gr.Markdown("Explore the capabilities of various Vision Language Models for tasks like OCR, VQA, and Object Detection.")

        with gr.Tabs():

            # --- TAB 1: Moondream3 Lab ---
            with gr.TabItem("🌝 Moondream3"):
                with gr.Row():
                    with gr.Column(scale=1):
                        md3_image_input = gr.Image(label="Upload an image", type="pil", height=400)
                        md3_task_type = gr.Radio(
                            choices=["Object Detection", "Point Detection", "Caption", "Visual Question Answering"],
                            label="Task Type", value="Object Detection"
                        )
                        md3_prompt_input = gr.Textbox(
                            label="Prompt (object to detect/question to ask)",
                            placeholder="e.g., 'car', 'person', 'What's in this image?'"
                        )
                        md3_max_objects = gr.Number(
                            label="Max Objects (for Object Detection only)",
                            value=10, minimum=1, maximum=50, step=1, visible=True
                        )
                        md3_generate_btn = gr.Button(value="Submit", variant="primary")
                    with gr.Column(scale=1):
                        md3_output_image = gr.Image(type="pil", label="Result", height=400)
                        md3_output_textbox = gr.Textbox(label="Model Response", lines=10, interactive=True)
                        md3_output_time = gr.Markdown()

                gr.Examples(
                    examples=[
                        ["md3/1.jpg", "Object Detection", "boats", 7],
                        ["md3/2.jpg", "Point Detection", "children", 7],
                        ["md3/3.png", "Caption", "", 5],
                        ["md3/4.jpeg", "Visual Question Answering", "Analyze the GDP trend over the years.", 5],
                    ],
                    inputs=[md3_image_input, md3_task_type, md3_prompt_input, md3_max_objects],
                    label="Click an example to populate inputs"
                )
                
            # --- TAB 2: Document and General VLMs ---
            with gr.TabItem("📄 Document & General VLM"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_choice = gr.Dropdown(
                            choices=["Camel-Doc-OCR-062825 (OCR)", "GLM-4.1V-9B (Thinking)"],
                            label="Select Model", value= "Camel-Doc-OCR-062825 (OCR)"
                        )
                        image_input_doc = gr.Image(label="Upload Image", type="pil", sources=['upload'], height=290)
                        prompt_input_doc = gr.Textbox(label="Query Input", placeholder="e.g., 'Transcribe the text in this document.'")
                        
                        with gr.Accordion("Advanced options", open=False):
                            max_new_tokens = gr.Slider(minimum=1, maximum=MAX_MAX_NEW_TOKENS, value=DEFAULT_MAX_NEW_TOKENS, step=1, label="Max New Tokens")
                            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.7)
                            top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, step=0.05, value=0.9)
                            top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=40)
                            repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)

                        with gr.Row():
                            process_btn = gr.Button("Submit", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")

                    with gr.Column(scale=3):
                        gr.Markdown("## Output", elem_id="output-title")
                        output_stream = gr.Textbox(label="Raw Output Stream", interactive=True, lines=24)
                            
                gr.Examples(
                    examples=[
                        ["examples/1.jpg", "Transcribe this receipt."],
                        ["examples/2.jpg", "Extract the content."],
                        ["examples/3.jpg", "OCR the image."],
                    ],
                    inputs=[image_input_doc, prompt_input_doc]
                )
            


        process_btn.click(
            fn=process_document_stream,
            inputs=[model_choice, image_input_doc, prompt_input_doc, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=[output_stream]
        )
        clear_btn.click(lambda: (None, "", ""), outputs=[image_input_doc, prompt_input_doc, output_stream])

        # Moondream3 Tab
        def update_max_objects_visibility(task):
            return gr.update(visible=(task == "Object Detection"))
        
        md3_task_type.change(fn=update_max_objects_visibility, inputs=[md3_task_type], outputs=[md3_max_objects])
        
        md3_generate_btn.click(
            fn=detect_objects_md3,
            inputs=[md3_image_input, md3_prompt_input, md3_task_type, md3_max_objects],
            outputs=[md3_output_image, md3_output_textbox, md3_output_time]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue(max_size=50).launch(theme=steel_blue_theme, css=css, ssr_mode=False, mcp_server=True, show_error=True)
