"""
FastAPI backend for 3D Asset Retrieval System (Qwen only).

Endpoints:
- GET /health - Health check
- POST /search/text - Search by text query
- POST /search/image - Search by image upload
"""
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from backend.embedding_service import EmbeddingService
from backend.qwen_db_bootstrap import ensure_qwen_db_ready
from backend.vector_search import VectorSearch
from utils.data_loader import DataLoader

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="3D Asset Retrieval API",
    description="Qwen-only multi-modal retrieval API for 3D assets",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService()
vector_search: Optional[VectorSearch] = None
data_loader = DataLoader()


class SearchResult(BaseModel):
    asset_id: str
    similarity: float
    caption_en: str
    caption_cn: Optional[str] = None
    objaverse_id: Optional[str] = None
    model_url: Optional[str] = None


class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Text query for search")
    language: str = Field("english", description="Query language: 'english' or 'chinese'")
    cross_modal: bool = Field(False, description="Enable cross-modal search (text->image)")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    high_quality_only: bool = Field(False, description="Filter for high quality assets only")


class TextSearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    algorithm: str
    language: str
    cross_modal: bool
    high_quality_only: bool


class ImageSearchResponse(BaseModel):
    results: List[SearchResult]
    algorithm: str
    cross_modal: bool
    high_quality_only: bool


@app.on_event("startup")
async def startup_event() -> None:
    global vector_search
    if config.QWEN_DB_AUTO_BOOTSTRAP:
        ensure_qwen_db_ready()
    vector_search = VectorSearch()
    logger.info("Asset retrieval backend started (Qwen-only)")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global vector_search
    logger.info("Shutting down...")
    if vector_search is not None:
        vector_search.close()
        vector_search = None


@app.get("/")
async def root():
    return {
        "name": "3D Asset Retrieval API",
        "version": "2.0.0",
        "status": "running",
        "algorithm": "qwen",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "algorithm": "qwen",
        "database": config.DB_NAME_QWEN,
        "qwen_db_auto_bootstrap": config.QWEN_DB_AUTO_BOOTSTRAP,
    }


@app.post("/search/text", response_model=TextSearchResponse)
async def search_text(request: TextSearchRequest):
    global vector_search
    if vector_search is None:
        raise HTTPException(status_code=503, detail="Service is not ready")

    if request.language not in ["english", "chinese"]:
        raise HTTPException(status_code=400, detail="Language must be 'english' or 'chinese'")

    try:
        logger.info(
            "Text search: query='%s...', language=%s, cross_modal=%s, hq=%s",
            request.query[:50],
            request.language,
            request.cross_modal,
            request.high_quality_only,
        )

        query_embedding = embedding_service.embed_query(query=request.query)
        results = vector_search.search(
            query_embedding=query_embedding,
            query_type="text",
            language=request.language,
            cross_modal=request.cross_modal,
            top_k=request.top_k,
            high_quality_only=request.high_quality_only,
        )

        for result in results:
            path_info = data_loader.get_objaverse_path_info(result["asset_id"])
            if path_info:
                result["objaverse_id"] = path_info["objaverse_id"]
                result["model_url"] = config.BASE_URL_TEMPLATE.format(**path_info)

        search_results = [SearchResult(**r) for r in results]
        return TextSearchResponse(
            results=search_results,
            query=request.query,
            algorithm="qwen",
            language=request.language,
            cross_modal=request.cross_modal,
            high_quality_only=request.high_quality_only,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=ImageSearchResponse)
async def search_image(
    file: UploadFile = File(...),
    cross_modal: bool = Query(False, description="Enable cross-modal search (image->text)"),
    language: str = Query("english", description="Target language for text-space search in cross-modal mode"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    high_quality_only: bool = Query(False, description="Filter for high quality assets only"),
):
    global vector_search
    if vector_search is None:
        raise HTTPException(status_code=503, detail="Service is not ready")

    if language not in ["english", "chinese"]:
        raise HTTPException(status_code=400, detail="Language must be 'english' or 'chinese'")

    try:
        logger.info(
            "Image search: filename=%s, cross_modal=%s, language=%s, hq=%s",
            file.filename,
            cross_modal,
            language,
            high_quality_only,
        )

        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        query_embedding = embedding_service.embed_query(image=image)
        results = vector_search.search(
            query_embedding=query_embedding,
            query_type="image",
            language=language,
            cross_modal=cross_modal,
            top_k=top_k,
            high_quality_only=high_quality_only,
        )

        for result in results:
            path_info = data_loader.get_objaverse_path_info(result["asset_id"])
            if path_info:
                result["objaverse_id"] = path_info["objaverse_id"]
                result["model_url"] = config.BASE_URL_TEMPLATE.format(**path_info)

        search_results = [SearchResult(**r) for r in results]
        return ImageSearchResponse(
            results=search_results,
            algorithm="qwen",
            cross_modal=cross_modal,
            high_quality_only=high_quality_only,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    logger.info("Starting FastAPI server on %s:%s", config.BACKEND_HOST, config.BACKEND_PORT)
    uvicorn.run(
        app,
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
