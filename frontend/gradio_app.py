"""
Gradio frontend for 3D Asset Retrieval System (Qwen only).
"""
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

API_BASE_URL = f"http://localhost:{config.BACKEND_PORT}"


def detect_language(text: str) -> str:
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    return "chinese" if chinese_pattern.search(text) else "english"


def format_results(results: List[dict], language: str = "english") -> str:
    html = "<div style='font-family: Arial, sans-serif;'>"
    html += "<h3>Search Results:</h3>"

    for i, result in enumerate(results, 1):
        similarity = result["similarity"]
        asset_id = result["asset_id"]
        caption = result.get("caption_cn" if language == "chinese" else "caption_en", "")
        color = "#00aa00" if similarity > 0.8 else "#aa8800" if similarity > 0.6 else "#aa0000"

        html += f"""
        <div style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9;'>
            <div style='font-weight: bold; color: {color};'>#{i} - Similarity: {similarity:.4f}</div>
            <div style='margin-top: 5px; color: #666;'>Asset ID: {asset_id}</div>
            <div style='margin-top: 5px;'>{caption}</div>
        </div>
        """

    html += "</div>"
    return html


def search_by_text(query: str, cross_modal: bool, top_k: int, high_quality_only: bool) -> Tuple[str, str, str]:
    try:
        if not query.strip():
            return None, "", "Please enter a search query", [], gr.Dropdown(choices=[], value=None)

        language = detect_language(query)
        response = requests.post(
            f"{API_BASE_URL}/search/text",
            json={
                "query": query,
                "language": language,
                "cross_modal": cross_modal,
                "top_k": top_k,
                "high_quality_only": high_quality_only,
            },
            timeout=60,
        )

        if response.status_code != 200:
            error_msg = response.json().get("detail", "Unknown error")
            return None, "", f"Search failed: {error_msg}", [], gr.Dropdown(choices=[], value=None)

        results_list = response.json().get("results", [])
        if not results_list:
            return None, "", "No results found", [], gr.Dropdown(choices=[], value=None)

        model_url = results_list[0].get("model_url")
        results_html = format_results(results_list, language)
        status_msg = f"Found {len(results_list)} results (Language: {language.title()}, Algorithm: Qwen, HQ: {high_quality_only})"

        choices = []
        for i, r in enumerate(results_list):
            caption = r.get("caption_en") or r.get("caption_cn") or r.get("asset_id", "Unknown")
            choices.append((f"#{i + 1}: {caption[:50]}...", i))

        return model_url, results_html, status_msg, results_list, gr.Dropdown(choices=choices, value=0)

    except requests.exceptions.ConnectionError:
        return None, "", "Error: Cannot connect to backend API.", [], gr.Dropdown(choices=[], value=None)
    except Exception as e:
        logger.error("Text search failed: %s", e, exc_info=True)
        return None, "", f"Error: {str(e)}", [], gr.Dropdown(choices=[], value=None)


def search_by_image(image, cross_modal: bool, top_k: int, high_quality_only: bool) -> Tuple[str, str, str]:
    try:
        if image is None:
            return None, "", "Please upload an image", [], gr.Dropdown(choices=[], value=None)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if hasattr(image, "save"):
                image.save(tmp.name)
            else:
                import shutil
                shutil.copy(image, tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            files = {"file": ("image.png", f, "image/png")}
            params = {
                "cross_modal": cross_modal,
                "top_k": top_k,
                "high_quality_only": high_quality_only,
            }
            response = requests.post(f"{API_BASE_URL}/search/image", files=files, params=params, timeout=60)

        Path(tmp_path).unlink(missing_ok=True)

        if response.status_code != 200:
            error_msg = response.json().get("detail", "Unknown error")
            return None, "", f"Search failed: {error_msg}", [], gr.Dropdown(choices=[], value=None)

        results_list = response.json().get("results", [])
        if not results_list:
            return None, "", "No results found", [], gr.Dropdown(choices=[], value=None)

        model_url = results_list[0].get("model_url")
        results_html = format_results(results_list, "english")
        status_msg = f"Found {len(results_list)} results (Algorithm: Qwen, HQ: {high_quality_only})"

        choices = []
        for i, r in enumerate(results_list):
            caption = r.get("caption_en") or r.get("caption_cn") or r.get("asset_id", "Unknown")
            choices.append((f"#{i + 1}: {caption[:50]}...", i))

        return model_url, results_html, status_msg, results_list, gr.Dropdown(choices=choices, value=0)

    except requests.exceptions.ConnectionError:
        return None, "", "Error: Cannot connect to backend API.", [], gr.Dropdown(choices=[], value=None)
    except Exception as e:
        logger.error("Image search failed: %s", e, exc_info=True)
        return None, "", f"Error: {str(e)}", [], gr.Dropdown(choices=[], value=None)


def create_ui():
    with gr.Blocks(title="3D Asset Retrieval(Objaverse Demo)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # 🎨 3D Asset Retrieval(Objaverse Demo)

        Search through millions of 3D assets using text or images with Qwen embeddings.

        **Features:**
        - 🔤 Text search in English and Chinese
        - 🖼️ Image-based search
        - 🔄 Cross-modal retrieval (text↔image)
        - 🤖 Qwen-only retrieval pipeline
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Search Options")

                cross_modal = gr.Checkbox(
                    label="Enable Cross-Modal Search",
                    value=False,
                    info="Search images with text or text with images",
                )

                high_quality_only = gr.Checkbox(
                    label="High Quality Only",
                    value=False,
                    info="Search only high quality assets",
                )

                top_k = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Number of Results")

                gr.Markdown("---")
                gr.Markdown("### 🔤 Search by Text")
                text_query = gr.Textbox(label="Text Query", placeholder="Enter description in English or Chinese...", lines=3)
                text_search_btn = gr.Button("🔍 Search by Text", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### 🖼️ Search by Image")
                image_query = gr.Image(label="Upload Image", type="pil")
                image_search_btn = gr.Button("🔍 Search by Image", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                status_msg = gr.Textbox(label="Status", interactive=False, lines=1)
                results_state = gr.State([])
                result_selector = gr.Dropdown(
                    label="Select Result to View in 3D",
                    choices=[],
                    type="index",
                    interactive=True,
                )
                model_3d = gr.Model3D(label="3D Model Viewer", height=400)
                results_display = gr.HTML(label="All Results")

        def update_model_from_dropdown(index, results):
            if results and index is not None and 0 <= index < len(results):
                return results[index].get("model_url")
            return None

        text_search_btn.click(
            fn=search_by_text,
            inputs=[text_query, cross_modal, top_k, high_quality_only],
            outputs=[model_3d, results_display, status_msg, results_state, result_selector],
        )

        image_search_btn.click(
            fn=search_by_image,
            inputs=[image_query, cross_modal, top_k, high_quality_only],
            outputs=[model_3d, results_display, status_msg, results_state, result_selector],
        )

        result_selector.change(
            fn=update_model_from_dropdown,
            inputs=[result_selector, results_state],
            outputs=[model_3d],
        )

    return demo


def main():
    logger.info("Starting Gradio app on %s:%s", config.FRONTEND_HOST, config.FRONTEND_PORT)
    demo = create_ui()
    demo.launch(server_name=config.FRONTEND_HOST, server_port=config.FRONTEND_PORT, share=True)


if __name__ == "__main__":
    main()
