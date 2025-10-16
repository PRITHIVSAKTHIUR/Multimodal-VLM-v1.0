# **Multimodal VLM v1.0**

A comprehensive multimodal vision-language model application supporting image inference and visual question answering. This repository hosts a Gradio-based demo that integrates several specialized models for document processing, OCR, spatial reasoning, and object/point detection.

<img width="1754" height="1171" alt="Screenshot 2025-10-16 at 12-00-45 Multimodal VLM v1 0 - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/737e6b41-0e3f-4d98-8e0f-9f002126f017" />

<img width="1771" height="1378" alt="Screenshot 2025-10-16 at 12-01-28 Multimodal VLM v1 0 - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/55b80cf5-cac7-42df-b112-044d70594a77" />

## Project Highlights

* **Interactive web UI** built with Gradio that supports streaming text output and image-based model tasks.
* **Three production-ready models** included in the demo:

  * `prithivMLmods/Camel-Doc-OCR-062825` — Qwen2.5-VL-based document OCR & structured extraction model.
  * `zai-org/GLM-4.1V-9B-Thinking` — GLM-4.1V visual reasoning model used for thinking-style multimodal queries.
  * `moondream/moondream3-preview` — lightweight model used for fast object detection, point detection, captioning, and VQA in the Moondream3 lab tab.
* **Streaming generation** using `TextIteratorStreamer` for real-time text display while the model generates.
* **On-device GPU support** via `spaces.GPU` decorator and automatic `torch` device detection.

## System Requirements

* **GPU**: Recommended NVIDIA GPU with CUDA support (demo auto-detects `cuda` vs `cpu`).
* **CUDA**: Compatible CUDA toolkit for your PyTorch build.
* **Python**: 3.10+
* **PyTorch**: A CUDA-enabled PyTorch build compatible with your GPU (e.g. `torch` with CUDA support).

> Note: The repository is developed and tested in an environment with modern CUDA and PyTorch builds. Exact version mismatch may cause model-loading errors; verify your installed `torch` version and CUDA compatibility.

## Required Python Packages

Install the core dependencies used by the demo:

```bash
pip install transformers gradio pillow numpy torch torchvision torchaudio supervision
# If you deploy on Hugging Face Spaces, also ensure `spaces` is available in that environment.
```

Additional optional tools referenced by the demo: `qwen-vl-utils` or other vendor utilities if you plan to use Qwen-specific processors outside of the provided `AutoProcessor`.

## Models Loaded in Code

The demo script (`app.py` or the main file) loads these models:

1. **Camel-Doc-OCR-062825** (`prithivMLmods/Camel-Doc-OCR-062825`)

   * Loaded with `AutoProcessor.from_pretrained` and `Qwen2_5_VLForConditionalGeneration.from_pretrained`.
   * Used for OCR/document extraction and structured markdown output.
   * `torch_dtype=torch.float16` and moved to the selected device.

2. **GLM-4.1V-9B-Thinking** (`zai-org/GLM-4.1V-9B-Thinking`)

   * Loaded with `AutoProcessor.from_pretrained` and `Glm4vForConditionalGeneration.from_pretrained`.
   * Used for visual reasoning tasks and multimodal Q&A.
   * `torch_dtype=torch.float16`.

3. **moondream3-preview** (`moondream/moondream3-preview`)

   * Loaded with `AutoModelForCausalLM.from_pretrained`.
   * The demo uses this model for quick image-based tasks: object detection, point detection, captioning, and VQA.
   * Configured with `torch_dtype=torch.bfloat16` and compiled on the device.

## How the Demo Works

### Key Functions

* `detect_objects_md3(image, prompt, task_type, max_objects)`

  * Runs Moondream3 model tasks with support for `Object Detection`, `Point Detection`, `Caption`, and `Visual Question Answering`.
  * Returns an annotated image, textual output, and inference timing.
  * Uses helper functions to draw bounding boxes (`create_annotated_image`) and point markers (`create_point_annotated_image`).

* `process_document_stream(model_name, image, prompt_input, ...)`

  * Handles Camel-Doc and GLM-4.1V streaming responses.
  * Uses `AutoProcessor` to prepare a chat-style prompt and `TextIteratorStreamer` to stream generation back to the UI.
  * Runs model generation in a background `Thread` to avoid blocking the Gradio interface.

### Gradio UI Layout

* **Tab 1 — Document & General VLM**

  * Dropdown to select `Camel-Doc-OCR-062825 (OCR)` or `GLM-4.1V-9B (Thinking)`.
  * Image upload, prompt input, advanced generation parameters (max_new_tokens, temperature, top-p, top-k, repetition_penalty).
  * Streaming text output in a large textbox.
  * Example inputs shipped in `examples/` to demonstrate OCR and document extraction.

* **Tab 2 — Moondream3**

  * Image upload, radio of task types (`Object Detection`, `Point Detection`, `Caption`, `Visual Question Answering`).
  * Prompt input used as detection label or question depending on the task.
  * Option to set `Max Objects` for object detection.
  * Annotated result image and model response text.

### UI Details

* Uses a custom CSS block and the `bethecloud/storj_theme` Gradio theme.
* Buttons configured with `elem_classes` for bespoke styling.
* `demo.queue(max_size=50)` to support concurrent requests in a queued fashion.
* Launch flags: `share=True, ssr_mode=False, mcp_server=True, show_error=True` (adjust these for your deployment environment).

## Usage Examples

* **OCR / Document Extraction**

  * Select `Camel-Doc-OCR-062825 (OCR)` and upload a scanned document image. Example prompt: `Transcribe this receipt.`

* **Visual Reasoning / VQA**

  * Select `GLM-4.1V-9B (Thinking)` and ask multimodal questions about the uploaded image.

* **Object Detection with Moondream3**

  * Switch to the `Moondream3` tab, select `Object Detection`, enter the object class (e.g. `car`) and set `Max Objects`.

---

## Advanced Configuration

* **Max Input Token Length** is set to `4096` in code. Long documents should be truncated or pre-split if you expect tokens > 4k.
* **Precision**: The script uses mixed precision (`float16` or `bfloat16`) when loading large models to reduce VRAM usage. Adjust `torch_dtype` if needed.
* **Device Mapping**: `moondream3-preview` is compiled and sent to CUDA with `device_map={"": "cuda"}` in the example code — change this if you are running on CPU or a different device topology.

## Performance & Limitations

* The demo is optimized for a single CUDA device but will fall back to CPU if no GPU is available.
* Streaming generation reduces perceived latency but model throughput depends heavily on GPU memory and compute.
* Objects/points annotation rely on normalized coordinates returned by the Moondream3 model — results may vary by model version.
* Example paths in the UI (`examples/` and `md3/`) assume those files exist in the repo; include them when packaging.

## Troubleshooting

* **Model Load Errors**: Ensure `trust_remote_code=True` for vendor models and that you have network access to download the model weights.
* **CUDA OOM**: Lower `max_new_tokens`, switch to smaller models, or use CPU-only debugging.
* **Slow or No Stream**: Confirm `TextIteratorStreamer` is compatible with the chosen model and that `processor.apply_chat_template` is returning the expected prompt.

## Contributing

Contributions, issues, and feature requests are welcome. Please open issues in the repository and include logs and reproduction steps for problems.

## License

This app is licensed under the **Apache License 2.0**. You may obtain a copy of the License at:

```
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
