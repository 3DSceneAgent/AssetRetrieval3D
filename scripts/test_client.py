"""
Backend test client for Qwen-only AssetRetrieval3D API.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BackendTestClient:
    def __init__(self, backend_url: str = "http://localhost:8002"):
        self.backend_url = backend_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> dict:
        response = self.session.get(f"{self.backend_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def search_text(
        self,
        query: str,
        language: str = "english",
        cross_modal: bool = False,
        top_k: int = 5,
        high_quality_only: bool = False,
    ) -> dict:
        payload = {
            "query": query,
            "language": language,
            "cross_modal": cross_modal,
            "top_k": top_k,
            "high_quality_only": high_quality_only,
        }
        response = self.session.post(f"{self.backend_url}/search/text", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def search_image(
        self,
        image_path: Path,
        cross_modal: bool = False,
        language: str = "english",
        top_k: int = 5,
        high_quality_only: bool = False,
    ) -> dict:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            params = {
                "cross_modal": cross_modal,
                "language": language,
                "top_k": top_k,
                "high_quality_only": high_quality_only,
            }
            response = self.session.post(
                f"{self.backend_url}/search/image",
                files=files,
                params=params,
                timeout=60,
            )
        response.raise_for_status()
        return response.json()


def _print_results(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    results = payload.get("results", [])
    print(f"results: {len(results)}")
    for i, item in enumerate(results[:5], 1):
        print(
            f"{i}. {item.get('asset_id')} | sim={item.get('similarity'):.4f} | "
            f"{(item.get('caption_en') or '')[:80]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-only backend smoke test client")
    parser.add_argument("--backend-url", default="http://localhost:8002")
    parser.add_argument("--query", default="a wooden chair")
    parser.add_argument("--language", choices=["english", "chinese"], default="english")
    parser.add_argument("--image", default=None, help="Optional local image path for image search")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cross-modal", action="store_true")
    parser.add_argument("--high-quality-only", action="store_true")
    args = parser.parse_args()

    client = BackendTestClient(args.backend_url)

    health = client.health_check()
    print("=== HEALTH ===")
    print(health)

    text_results = client.search_text(
        query=args.query,
        language=args.language,
        cross_modal=args.cross_modal,
        top_k=args.top_k,
        high_quality_only=args.high_quality_only,
    )
    _print_results("TEXT SEARCH", text_results)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_results = client.search_image(
            image_path=image_path,
            cross_modal=args.cross_modal,
            language=args.language,
            top_k=args.top_k,
            high_quality_only=args.high_quality_only,
        )
        _print_results("IMAGE SEARCH", image_results)


if __name__ == "__main__":
    main()
