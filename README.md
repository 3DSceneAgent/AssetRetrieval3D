# 3D Asset Retrieval Demo

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange.svg)](https://17d9a08e2e8b57de04.gradio.live)
[![中文](https://img.shields.io/badge/中文-README-blue.svg)](README_zh.md)

A simple multi-modal 3D asset retrieval system. This repo uses objaverse as a simple demonstration, while it can generalize and scale to any 3D dataset.

![Demo](./assets/asset_retrieval_demo.gif)

## Project Status Update

- Runtime/API path is now **Qwen-only** in this repo (**SigLIP deprecated**).
- Backend supports startup-time DB bootstrap from OSS via `QWEN_DB_OSS_URL`.

## 1. Features
> *Demo built upon objaverse, using Cap3D 650k+ english captions, 650k+  translated captions ,and 260k+ gobjaverse asset renderings*
- **Text Search**: Retrieve 3D assets in English or Chinese
- **Image Search**: Retrieve 3D assets using a single RGB image
- **Cross-Modal Retrieval**: Retrieve 3D assets using text2image or image2text similarities
- **Qwen3-VL-Embedding**: Bilingual, *multi-image* embeddings. High Retrieval Quality.
- **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **Web Interface**: Beautiful Gradio UI with 3D model viewer
- **REST API**: FastAPI backend for programmatic access

## 2. Architecture

```
┌─────────────────┐
│ English Captions│
└────────┬────────┘
         │
         ├──► Translation ──► Chinese Captions
         │
         └──► Qwen Embeddings ─────┤
              - Text (EN + CN)     │
              - Images (multi)     │
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ PostgreSQL       │
                         │ + pgvector       │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  FastAPI Backend │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Gradio Frontend │
                         └──────────────────┘
```



## 3. Get Started
### Prerequisites
- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- DashScope API key (for Qwen)
### Installation
1. **Clone the repository** 
```bash 
git clone https://github.com/3DSceneAgent/AssetRetrieval3D
```

2. **Install dependencies**:
   ```bash
   # optionally create new env 
   # conda create -n asset_retrieval python=3.10
   # conda activate asset_retrieval
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env 
   # setup variables
   export DASHSCOPE_API_KEY="your-api-key"
   export DB_HOST="localhost"
   export DB_PORT="5432"
   export DB_USER="postgres"
   export DB_PASSWORD="your-password"
   ```

4. **Configure PostgreSQL**:
   - Install PostgreSQL
   - Install pgvector extension:
     ```sql
     CREATE EXTENSION vector;
     ```

## 4. Usage

### Step 0: Configure Settings

Edit `config.py` to adjust settings:
- `MAX_ASSETS`: Set to a small number (e.g., 100) for testing
- Database credentials
- API keys
- Paths

### Step 1: Translate Captions (Optional but Recommended)

Translate English captions to Chinese:

```bash
python scripts/01_translate_captions.py
```

This creates `data/text_captions_cap3d_cn.json`.

**Note**: This uses the Qwen batch API and may take hours for the full dataset.

### Step 2: Generate Embeddings

```bash
python scripts/03_embed_qwen.py
```

This generates:
- Text embeddings (English and Chinese)
- Multi-image embeddings (8 images per asset)

**Note**: Can run both in parallel if you have resources.

### Step 3: Populate Database

```bash
python scripts/04_populate_database.py
```

This:
- Creates database: `qwen_embeddings`
- Creates tables with pgvector columns
- Inserts all embeddings
- Creates vector indexes

### Step 4: Start Backend API

```bash
python backend/app.py
```

Or with uvicorn:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8002
```

The API will be available at `http://localhost:8002`

### Step 5: Start Frontend

In a new terminal:

```bash
python frontend/gradio_app.py
```

The UI will be available at `http://localhost:7860`

### Step 6: Test Backend with Client Script

A comprehensive test client is provided to verify the backend service:

```bash
# Run all tests with default settings
python test_client.py

# Test only image search
python test_client.py --test-type image

# Run with verbose output (shows all results)
python test_client.py --verbose --num-tests 1

# Test against remote backend
python test_client.py --backend-url http://remote-server:8002
```

The test client will:
- Check backend health
- Test text search (English and Chinese for Qwen)
- Test image search using real assets from the database
- Test both uni-modal and cross-modal search
- Display results with similarity scores and captions
 /search/text


## 5. Configuration Options

### `config.py` Key Settings

- **MAX_ASSETS**: Limit number of assets to process (for debugging)
- **TRANSLATION_BATCH_SIZE**: Captions per translation batch (default: 1000)
- **EMBEDDING_BATCH_SIZE**: Batch size for embedding generation (default: 100)
- **QWEN_NUM_IMAGES**: Number of images per asset for Qwen (default: 8)
- **DEFAULT_TOP_K**: Default number of search results (default: 10)

## 6. Algorithms
#### Inner-Modal Search
- **Text → Text**: Find assets with similar descriptions
- **Image → Image**: Find visually similar assets
#### Cross-Modal Search
- **Text → Image**: Find images matching text description
- **Image → Text**: Find text descriptions matching image


## 7. File Structure

```
objaverse_retrieval/
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── README.md                # This file
├── test_client.py           # Backend test client
│
├── data/                    # Data files
│   ├── text_captions_cap3d.json
│   ├── text_captions_cap3d_cn.json  (generated)
│   ├── gobjaverse/          # 3D asset images
│   └── gobjaverse_280k_index_to_objaverse.json
│
├── outputs/                 # Generated outputs
│   ├── embeddings/          # Saved embeddings
│   ├── translations/        # Translation results
│   └── batch_jsonl/         # API batch files
│
├── utils/                   # Utility modules
│   ├── data_loader.py
│   ├── image_utils.py
│   └── db_utils.py
│
├── scripts/                 # Processing scripts
│   ├── 01_translate_captions.py
│   ├── 03_embed_qwen.py
│   └── 04_populate_database.py
│
├── backend/                 # FastAPI backend
│   ├── app.py
│   ├── embedding_service.py
│   └── vector_search.py
│
└── frontend/               # Gradio frontend
    └── gradio_app.py
```

## License
The code and application is licensed under [Apache2.0 License](LICENSE).

## Acknowledgments
* [objaverse](https://objaverse.allenai.org/)
* [objaverse_filter from kiui](https://github.com/ashawkey/objaverse_filter)
* [gobjaverse](https://github.com/modelscope/richdreamer)
* [Cap3D](https://github.com/crockwell/Cap3D/)
