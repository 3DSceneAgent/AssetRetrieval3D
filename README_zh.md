# 3D 资产检索 Demo

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange.svg)](https://17d9a08e2e8b57de04.gradio.live)
[![English](https://img.shields.io/badge/English-README-blue.svg)](README.md)

多模态3D资产检索系统，本项目使用 Objaverse 作为 Demo演示，但框架可以扩展到任意3D数据库。

![Demo](./assets/asset_retrieval_demo.gif)

## 项目状态更新

- 当前仓库运行链路已调整为 **Qwen-only**(SigLIP已废弃）。
- 后端支持通过 `QWEN_DB_OSS_URL` 在启动时从 OSS 自动恢复数据库。

## 1. 功能特性
> *基于Cap3D提供的65w+英文标题，qwen翻译后的65w+中文标题，gobjaverse得到的数百万张图片渲染得到嵌入*
- **文本搜索**：支持用英文或中文检索3D资产
- **图像搜索**：使用单张RGB图像检索3D 资产
- **跨模态检索**：以文图相似度或以图文相似度检索3D资产。
- **双算法支持**：
  - **SigLip**：快速，仅支持英文，对3D资产每个视角的渲染图生成一个Embedding。检索质量中等(实现中)。
  - **Qwen3-VL-Embedding**：双语嵌入，联合多图进行Embedding。检索质量高。
- **向量数据库**：使用 PostgreSQL 和 pgvector 进行高效的相似度搜索
- **Web 界面**：基于 Gradio 的美观 UI，包含 3D 模型查看器
- **REST API**：用于程序化访问的 FastAPI 后端

## 2. 架构

```
┌─────────────────┐
│     英文描述     │
└────────┬────────┘
         │
         ├──► 翻译 ────────► 中文描述
         │
         ├──► SigLip Embeddings ──┐
         │    - 文本 (EN)          │
         │    - 图像 (单视角)       │
         │                         │
         └──► Qwen Embeddings ─────┤
              - 文本 (EN + CN)     │
              - 图像 (多视角)       │
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

## 3. 快速开始

### 前提条件
- Python 3.8+
- PostgreSQL 12+ (需安装 pgvector 扩展)
- NVIDIA GPU (SigLip 推荐)
- DashScope API key (Qwen 需要)

### 安装步骤
1. **克隆仓库**
```bash 
git clone https://github.com/3DSceneAgent/AssetRetrieval3D
```

2. **安装依赖**:
   ```bash
   # 创建conda环境(可选)
   conda create -n asset_retrieval python=3.10
   conda activate asset_retrieval
   pip install -r requirements.txt
   ```

3. **设置环境变量**:
   ```bash
   cp .env.example .env
   export DASHSCOPE_API_KEY="your-api-key"
   export DB_HOST="localhost"
   export DB_PORT="5432"
   export DB_USER="postgres"
   export DB_PASSWORD="your-password"
   ```

4. **配置 PostgreSQL**:
   - 安装 PostgreSQL
   - 安装 pgvector 扩展:
     ```sql
     CREATE EXTENSION vector;
     ```

## 4. 使用方法

### 第 0 步: 配置设置

编辑 `config.py` 以调整设置：
- `MAX_ASSETS`：设置为较小的数字（例如 100）以便进行测试
- 数据库凭据
- API 密钥
- 路径配置

### 第 1 步: 翻译描述 (可选但推荐)

将英文描述翻译为中文：

```bash
python scripts/01_translate_captions.py
```

这将创建 `data/text_captions_cap3d_cn.json`。

**注意**：此步骤使用 Qwen 批量 API，完整数据集可能需要数小时。

### 第 2 步: 生成 Embeddings

#### (可选) SigLip Embeddings

```bash
python scripts/02_embed_siglip.py
```

这将生成：
- 文本嵌入 (仅英文)
- 图像嵌入 (每个视角一个)

#### (可选) Qwen Embeddings

```bash
python scripts/03_embed_qwen.py
```

这将生成：
- 文本嵌入 (英文和中文)
- 多图像嵌入 (每个资产 8 张图像)

**注意**：如果有足够的资源，可以并行运行这两个脚本。

### 第 3 步: 填充数据库

```bash
python scripts/04_populate_database.py
```

此步骤将：
- 创建两个数据库：`siglip_embeddings` 和 `qwen_embeddings`
- 创建带有 pgvector 列的表
- 插入所有 embeddings
- 创建向量索引

### 第 4 步: 启动后端 API

```bash
python backend/app.py
```

或者使用 uvicorn：
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

API 将在 `http://localhost:8000` 可用。

### 第 5 步: 启动前端

在新的终端中运行：

```bash
python frontend/gradio_app.py
```

UI 将在 `http://localhost:7860` 可用。

### 第 6 步: 使用客户端脚本测试后端

提供了一个综合测试客户端来验证后端服务：

```bash
# 使用默认设置运行所有测试
python test_client.py

# 仅测试 SigLip 算法
python test_client.py --algorithm siglip

# 仅测试图像搜索
python test_client.py --test-type image

# 运行详细输出模式 (显示所有结果)
python test_client.py --verbose --num-tests 1

# 针对远程后端进行测试
python test_client.py --backend-url http://remote-server:8000
```

测试客户端将：
- 检查后端运行状况
- 测试文本搜索 (Qwen 支持中英文)
- 使用数据库中的真实资产测试图像搜索
- 测试单模态和跨模态搜索
- 显示带有相似度分数和描述的结果

## 5. 配置选项

### `config.py` 关键设置

- **MAX_ASSETS**: 限制处理的资产数量 (用于调试)
- **TRANSLATION_BATCH_SIZE**: 每次翻译批次的描述数量 (默认: 1000)
- **EMBEDDING_BATCH_SIZE**: 生成 embedding 的批次大小 (默认: 100)
- **QWEN_NUM_IMAGES**: Qwen 每个资产使用的图像数量 (默认: 8)
- **DEFAULT_TOP_K**: 默认搜索结果数量 (默认: 10)

## 6. 算法

### Embedding 算法
| 特性 | SigLip | Qwen |
|---------|--------|------|
| 文本语言 | 仅英文 | 英文 + 中文 |
| 图像 Embeddings | 每个视角一个 | 多图像 (8 视角) |
| 速度 | 快 | 较慢 (API 调用) |
| 需要 GPU | 是 (本地) | 否 (API) |
| 跨模态 | 支持 | 支持 |

### 搜索模式
#### 模态内搜索 (Inner-Modal)
- **文搜文**: 查找具有相似描述的资产
- **图搜图**: 查找视觉上相似的资产
#### 跨模态搜索 (Cross-Modal)
- **文搜图**: 查找匹配文本描述的图像
- **图搜文**: 查找匹配图像的文本描述

## 7. 文件结构

```
objaverse_retrieval/
├── config.py                 # 配置文件
├── requirements.txt          # 依赖项
├── README.md                # 说明文件 (英文)
├── README_zh.md             # 说明文件 (中文)
├── test_client.py           # 后端测试客户端
│
├── data/                    # 数据文件
│   ├── text_captions_cap3d.json
│   ├── text_captions_cap3d_cn.json  (已生成)
│   ├── gobjaverse/          # 3D 资产图像
│   └── gobjaverse_280k_index_to_objaverse.json
│
├── outputs/                 # 生成的输出
│   ├── embeddings/          # 保存的 embeddings
│   ├── translations/        # 翻译结果
│   └── batch_jsonl/         # API 批处理文件
│
├── utils/                   # 工具模块
│   ├── data_loader.py
│   ├── image_utils.py
│   └── db_utils.py
│
├── scripts/                 # 处理脚本
│   ├── 01_translate_captions.py
│   ├── 02_embed_siglip.py
│   ├── 03_embed_qwen.py
│   └── 04_populate_database.py
│
├── backend/                 # FastAPI 后端
│   ├── app.py
│   ├── embedding_service.py
│   └── vector_search.py
│
└── frontend/               # Gradio 前端
    └── gradio_app.py
```

## 许可证
[Apache2.0 License](LICENSE)

## 致谢
* [objaverse](https://objaverse.allenai.org/)
* [objaverse_filter from kiui](https://github.com/ashawkey/objaverse_filter)
* [gobjaverse](https://github.com/modelscope/richdreamer)
* [Cap3D](https://github.com/crockwell/Cap3D/)
