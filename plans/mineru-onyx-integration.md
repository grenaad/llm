# MinerU + Onyx Integration Plan

## Goal

Add MinerU as a sidecar service in the Onyx docker-compose stack to enable OCR
and advanced document parsing (scanned PDFs, photos, digital PDFs, DOCX) with
the parsed output automatically pushed to Onyx for search and RAG chat.

## Architecture

```
[Browser]
    |
    |--- http://localhost:7002  --> [Onyx UI]        (search + chat)
    |--- http://localhost:7860  --> [MinerU Gradio]   (document upload + OCR)

[MCP Clients: Claude Code, Cursor, Windsurf, etc.]
    |
    |--- http://localhost:8090  --> [Onyx MCP Server] (search_indexed_documents)

                     Inside Docker network "onyx"
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  [MinerU Gradio + API]  ──parse──>  [Markdown/JSON output]      │
    │       (GPU: 1080 Ti)                       │                    │
    │                                            │                    │
    │                      [push_to_onyx.py]                          │
    │                        │              │                         │
    │          (first run)   │              │  (every document)       │
    │     Create File        │              │  POST /onyx-api/        │
    │     Connector via API  │              │       ingestion         │
    │                        ▼              ▼                         │
    │                                                                 │
    │                        [Onyx api_server :8080]                  │
    │                                │                                │
    │                        [Vespa index + embeddings]               │
    │                           (GPU: 1060)                           │
    │                                │                                │
    │                        [Onyx MCP Server :8090]                  │
    │                                │                                │
    │                   search_indexed_documents                      │
    │                   search_web                                    │
    │                   open_urls                                     │
    └─────────────────────────────────────────────────────────────────┘
```

## Key Constraints

| Constraint | Detail |
|---|---|
| GPU 0: GTX 1080 Ti | 11 GB VRAM, Compute Capability 6.1 (Pascal). Used by Whisper. |
| GPU 1: GTX 1060 3GB | 3 GB VRAM, CC 6.1 (Pascal). Used by Onyx embeddings. |
| System RAM | 15.6 GB total, ~9 GB available after Whisper + MetaTrader |
| MinerU official Docker image | Based on `vllm/vllm-openai:v0.11.2` which requires CC 7.0+ (Volta). **Will NOT work** on our Pascal GPUs. |
| MinerU pipeline backend | Uses PyTorch + ONNX models. Works on CC 6.1. Scores 86.2 on OmniDocBench v1.5. |
| MinerU VLM engine | Requires vLLM, which requires CC 7.0+. **Cannot use on our hardware.** |
| Onyx auth | Currently disabled (`AUTH_TYPE=disabled`). Ingestion API still needs Bearer token. |
| Onyx host port | 7002 (configured in `.env` as `HOST_PORT=7002`) |

## Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| MinerU inference backend | `pipeline` only (no VLM) | Pascal GPUs can't run vLLM; pipeline is sufficient for our doc types |
| Docker image | Custom build from PyTorch base | Official image requires CC 7.0+; we need CC 6.1 support |
| GPU assignment | GPU 0 (1080 Ti) | Has 11 GB VRAM (pipeline needs ~3-4 GB); shares with Whisper but rarely concurrent |
| UI approach | Extend MinerU's built-in Gradio | Minimal custom code; already supports all target file types |
| Onyx integration | Ingestion API (`POST /onyx-api/ingestion`) | Lightweight, documented, supports sections + metadata |
| Deployment | Sidecar in Onyx compose (overlay file) | Shares Docker network for internal API calls; modular via compose overlay |
| File Connector | Auto-create via API on first run | No manual Admin Panel steps; `cc_pair_id` needed for docs to appear in Connectors page and filter as `"file"` source in MCP |
| MCP access | Enabled (`MCP_SERVER_ENABLED=true`, port 8090) | Documents ingested via MinerU are automatically searchable by Claude Code, Cursor, etc. |
| OpenSearch | Disable before first Onyx run | Saves ~2 GB RAM; not needed for small/personal project |

---

## Implementation Steps

### Phase 1: Prepare Onyx Environment

#### 1.1 Disable OpenSearch in Onyx `.env`

**File**: `onyx_data/deployment/.env`

Change:
```
OPENSEARCH_FOR_ONYX_ENABLED=true
```
To:
```
OPENSEARCH_FOR_ONYX_ENABLED=false
```

Also remove `opensearch` from `COMPOSE_PROFILES` if present. This saves ~2 GB RAM.

#### 1.2 Start Onyx (first time)

```bash
cd onyx_data/deployment
make run-gpu
```

Wait for all services to be healthy. Access Onyx at `http://localhost:7002`.

#### 1.3 Create an Onyx API Key

1. Open Onyx at `http://localhost:7002`
2. Navigate to **Admin > API Keys**
3. Create a new **Admin API Key**
4. Copy the key and add it to `.env`:
   ```
   ONYX_API_KEY=<your_api_key_here>
   ```

This is the only manual step. The File Connector and `cc_pair_id` are created
automatically by `push_to_onyx.py` on its first run (see Phase 2.3).

#### 1.4 Verify MCP server is running

The MCP server is already enabled in `.env` (`MCP_SERVER_ENABLED=true`).
Verify it's healthy:

```bash
curl http://localhost:8090/health
# Expected: {"status": "healthy", "service": "mcp_server"}
```

---

### Phase 2: Build Custom MinerU Docker Image

#### 2.1 Create directory structure

```
onyx_data/deployment/mineru/
├── Dockerfile
├── push_to_onyx.py          # Post-parse hook script
├── mineru_with_onyx.py       # Custom Gradio app wrapping MinerU + Onyx push
└── requirements-extra.txt    # Additional Python deps (httpx for Onyx API calls)
```

#### 2.2 Write the Dockerfile

**File**: `onyx_data/deployment/mineru/Dockerfile`

```dockerfile
# Custom MinerU image for Pascal GPUs (CC 6.1)
# Based on PyTorch with CUDA 11.8 (supports CC 3.5 - 9.0)
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Set non-interactive to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - libgl1: required by OpenCV
# - fonts-noto-*: CJK font support for document rendering
# - poppler-utils: PDF utilities (pdfinfo, pdftotext)
# - curl: healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        curl && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install MinerU with pipeline + gradio backends (no vLLM/VLM)
# Also install httpx for Onyx API integration
RUN pip install --no-cache-dir \
    'mineru[pipeline,gradio]>=3.0.0' \
    httpx

# Download pipeline models only (not VLM models)
# This downloads: layout detection, OCR, table recognition, formula models
RUN mineru-models-download -s huggingface -m pipeline

# Copy custom integration scripts
COPY push_to_onyx.py /app/push_to_onyx.py
COPY mineru_with_onyx.py /app/mineru_with_onyx.py

WORKDIR /app

# Environment defaults
ENV MINERU_MODEL_SOURCE=local
ENV ONYX_API_URL=http://api_server:8080/api
ENV ONYX_API_KEY=""
# ONYX_CC_PAIR_ID is auto-managed by push_to_onyx.py (cached in output volume)

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default: run the custom Gradio app that wraps MinerU + Onyx push
ENTRYPOINT ["python", "/app/mineru_with_onyx.py"]
CMD ["--server-name", "0.0.0.0", "--server-port", "7860"]
```

**Key decisions in this Dockerfile:**
- `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime` supports CC 6.1 (Pascal)
- Only install `mineru[pipeline,gradio]` — no VLM/vLLM dependencies
- Only download `pipeline` models (~2-3 GB) — not the VLM model (~3+ GB)
- Custom scripts for Onyx integration are copied in

**Estimated image size**: ~8-10 GB (PyTorch ~4 GB, models ~3 GB, deps ~2 GB)

#### 2.3 Write the Onyx push script

**File**: `onyx_data/deployment/mineru/push_to_onyx.py`

This module:
1. **Auto-creates a File Connector** on first run via Onyx's Admin API
   (no manual Admin Panel steps needed)
2. Caches the `cc_pair_id` to a local file so it persists across restarts
3. Pushes parsed Markdown content to Onyx's Ingestion API

```python
"""
push_to_onyx.py - Push MinerU parsed documents to Onyx Ingestion API

On first run, automatically creates a File Connector in Onyx via the Admin API.
Subsequent runs reuse the cached cc_pair_id.

Onyx APIs used:
  - POST /manage/admin/connector          (create connector)
  - PUT  /manage/admin/connector/{id}/credential/0  (associate empty credential)
  - GET  /manage/admin/connector/indexing-status     (find existing connector)
  - POST /onyx-api/ingestion              (push documents)

Docs: https://docs.onyx.app/developers/guides/index_files_ingestion_api
      https://docs.onyx.app/developers/guides/create_connector
"""

import os
import re
import json
import logging
import httpx
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ONYX_API_URL = os.environ.get("ONYX_API_URL", "http://api_server:8080/api")
ONYX_API_KEY = os.environ.get("ONYX_API_KEY", "")
CC_PAIR_CACHE_FILE = Path("/app/output/.onyx_cc_pair_id")

CONNECTOR_NAME = "mineru-ocr-documents"

# Module-level cache
_cc_pair_id: int | None = None


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {ONYX_API_KEY}",
        "Content-Type": "application/json",
    }


def _ensure_file_connector() -> int | None:
    """
    Ensure a File Connector named 'mineru-ocr-documents' exists in Onyx.
    Creates one via API if it doesn't exist. Returns the cc_pair_id.

    Uses a 3-step process:
      1. Check if connector already exists (GET indexing-status)
      2. If not, create connector (POST /manage/admin/connector)
      3. Associate with empty credential (PUT connector/{id}/credential/0)

    The cc_pair_id is cached to /app/output/.onyx_cc_pair_id for persistence
    across container restarts.
    """
    global _cc_pair_id

    # Return cached value if available
    if _cc_pair_id is not None:
        return _cc_pair_id

    # Try reading from cache file
    if CC_PAIR_CACHE_FILE.exists():
        try:
            _cc_pair_id = int(CC_PAIR_CACHE_FILE.read_text().strip())
            logger.info(f"Loaded cached cc_pair_id: {_cc_pair_id}")
            return _cc_pair_id
        except (ValueError, OSError):
            pass

    if not ONYX_API_KEY:
        logger.warning("ONYX_API_KEY not set, skipping connector creation")
        return None

    try:
        with httpx.Client(timeout=30.0) as client:
            # Step 1: Check if our connector already exists
            resp = client.get(
                f"{ONYX_API_URL}/manage/admin/connector/indexing-status",
                headers=_headers(),
            )
            resp.raise_for_status()
            for item in resp.json():
                connector = item.get("connector", {})
                if connector.get("name") == CONNECTOR_NAME:
                    _cc_pair_id = item.get("cc_pair_id")
                    if _cc_pair_id:
                        logger.info(
                            f"Found existing connector "
                            f"'{CONNECTOR_NAME}': cc_pair_id={_cc_pair_id}"
                        )
                        CC_PAIR_CACHE_FILE.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        CC_PAIR_CACHE_FILE.write_text(str(_cc_pair_id))
                        return _cc_pair_id

            # Step 2: Create the connector
            logger.info(f"Creating File Connector '{CONNECTOR_NAME}'...")
            resp = client.post(
                f"{ONYX_API_URL}/manage/admin/connector",
                headers=_headers(),
                json={
                    "name": CONNECTOR_NAME,
                    "source": "file",
                    "input_type": "poll",
                    "access_type": "PUBLIC",
                    "connector_specific_config": {},
                    "refresh_freq": None,
                    "prune_freq": None,
                },
            )
            resp.raise_for_status()
            connector_id = resp.json()["id"]
            logger.info(f"Created connector id={connector_id}")

            # Step 3: Associate with empty credential (id=0)
            # File connectors don't need real credentials
            resp = client.put(
                f"{ONYX_API_URL}/manage/admin/connector/"
                f"{connector_id}/credential/0",
                headers=_headers(),
            )
            resp.raise_for_status()
            result = resp.json()

            # The response contains the cc_pair_id
            _cc_pair_id = result.get("data", {}).get("cc_pair_id")
            if not _cc_pair_id:
                # Fallback: re-query indexing status to find it
                resp = client.get(
                    f"{ONYX_API_URL}/manage/admin/connector/indexing-status",
                    headers=_headers(),
                )
                resp.raise_for_status()
                for item in resp.json():
                    connector = item.get("connector", {})
                    if connector.get("name") == CONNECTOR_NAME:
                        _cc_pair_id = item.get("cc_pair_id")
                        break

            if _cc_pair_id:
                CC_PAIR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                CC_PAIR_CACHE_FILE.write_text(str(_cc_pair_id))
                logger.info(
                    f"File Connector ready: cc_pair_id={_cc_pair_id}"
                )
            else:
                logger.warning(
                    "Connector created but could not determine cc_pair_id"
                )

            return _cc_pair_id

    except Exception as e:
        logger.error(f"Failed to create File Connector: {e}")
        return None


def push_to_onyx(
    filename: str,
    markdown_content: str,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Push parsed document content to Onyx's Ingestion API.

    On first call, ensures a File Connector exists (auto-created via API).
    Documents are associated with the connector so they appear in the
    Admin > Connectors page and are filterable as source "file" in MCP.

    Args:
        filename: Original document filename (used as semantic_identifier)
        markdown_content: The parsed Markdown text from MinerU
        metadata: Optional metadata dict (file_type, page_count, etc.)

    Returns:
        dict with keys: success, document_id, already_existed, error
    """
    if not ONYX_API_KEY:
        return {
            "success": False,
            "document_id": None,
            "error": "ONYX_API_KEY not configured",
        }

    # Ensure File Connector exists (auto-creates on first call)
    cc_pair_id = _ensure_file_connector()

    # Build the ingestion payload
    sections = _split_into_sections(markdown_content)

    payload = {
        "document": {
            "semantic_identifier": filename,
            "sections": sections,
            "source": "file",
            "metadata": {
                "parsed_by": "mineru",
                "file_type": Path(filename).suffix.lower(),
                **(metadata or {}),
            },
            "from_ingestion_api": True,
        },
    }

    if cc_pair_id:
        payload["cc_pair_id"] = cc_pair_id

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{ONYX_API_URL}/onyx-api/ingestion",
                json=payload,
                headers=_headers(),
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                f"Pushed to Onyx: {filename} -> "
                f"doc_id={result.get('document_id')}, "
                f"existed={result.get('already_existed')}"
            )
            return {
                "success": True,
                "document_id": result.get("document_id"),
                "already_existed": result.get("already_existed", False),
                "error": None,
            }
    except httpx.HTTPStatusError as e:
        error_msg = (
            f"Onyx API error {e.response.status_code}: {e.response.text}"
        )
        logger.error(error_msg)
        return {"success": False, "document_id": None, "error": error_msg}
    except Exception as e:
        error_msg = f"Failed to push to Onyx: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "document_id": None, "error": error_msg}


def _split_into_sections(markdown: str) -> list[dict]:
    """
    Split markdown content into sections for Onyx ingestion.

    Splits on page breaks (---) or major headers (## / #).
    Each section becomes a separate searchable chunk in Onyx.
    """
    if not markdown.strip():
        return [{"text": "(empty document)"}]

    # Try splitting on page breaks first (MinerU uses \n---\n separators)
    pages = re.split(r"\n-{3,}\n", markdown)

    if len(pages) <= 1:
        # No page breaks found, split on major headers
        parts = re.split(r"\n(?=#{1,2}\s)", markdown)
        if len(parts) <= 1:
            return [{"text": markdown.strip()}]
        return [{"text": part.strip()} for part in parts if part.strip()]

    return [{"text": page.strip()} for page in pages if page.strip()]
```

**Key design points:**
- Connector is auto-created on first `push_to_onyx()` call — no manual setup
- `cc_pair_id` is cached to `/app/output/.onyx_cc_pair_id` (inside the
  `mineru_output` Docker volume) so it persists across container restarts
- `credential_id: 0` is Onyx's built-in empty credential for File/Web
  connectors that don't need auth
- Documents use `source: "file"` so they appear as `"file"` type in MCP's
  `source_types` filter, not `"ingestion_api"`

#### 2.4 Write the custom Gradio wrapper

**File**: `onyx_data/deployment/mineru/mineru_with_onyx.py`

This is the main entry point. It wraps MinerU's standard Gradio app and adds an
"auto-push to Onyx" toggle/callback. There are two approaches:

**Approach A (preferred)**: Import and extend MinerU's Gradio app, adding a
post-processing callback that calls `push_to_onyx()` after parsing completes.

**Approach B (fallback)**: Run `mineru-gradio` as-is, plus a separate background
watcher that monitors the output directory and pushes new results to Onyx.

The exact implementation depends on MinerU's Gradio app internals — we need to
inspect the source to see if it exposes hooks or events. The implementation step
should:

1. Import MinerU's Gradio app module
2. Identify the parse completion event/callback
3. Add a Gradio checkbox: "Auto-push to Onyx" (default: on)
4. Add a Gradio textbox showing push status ("Pushed to Onyx: doc_id=xxx")
5. On parse complete, if checkbox is checked, call `push_to_onyx()`

**Research needed during implementation:**
- Inspect `mineru.cli.gradio_app` source to find the Gradio app object and
  parse callback
- Check if MinerU's Gradio app returns the parsed Markdown as a string we can
  intercept
- If the Gradio app isn't easily extensible, fall back to Approach B (output
  dir watcher)

---

### Phase 3: Add MinerU to Onyx Docker Compose

#### 3.1 Create compose overlay file

**File**: `onyx_data/deployment/docker-compose.mineru.yml`

```yaml
# MinerU OCR sidecar for Onyx
# Usage:
#   docker compose -f docker-compose.yml \
#     -f docker-compose.gpu.yml \
#     -f docker-compose.mineru.yml up -d

services:
  mineru:
    build:
      context: ./mineru
      dockerfile: Dockerfile
    container_name: onyx-mineru
    restart: unless-stopped
    ports:
      - "7860:7860"
    environment:
      MINERU_MODEL_SOURCE: local
      ONYX_API_URL: http://api_server:8080/api
      ONYX_API_KEY: ${ONYX_API_KEY:-}
      # cc_pair_id is auto-created and cached by push_to_onyx.py
    volumes:
      - mineru_output:/app/output
      - mineru_model_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # GTX 1080 Ti (CC 6.1, 11 GB)
              capabilities: [gpu]
    depends_on:
      api_server:
        condition: service_started
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host

volumes:
  mineru_output:
  mineru_model_cache:
```

**Notes:**
- `depends_on: api_server` ensures Onyx API is up before MinerU starts pushing
- `mineru_model_cache` prevents re-downloading models on container recreate
- `ipc: host` and `ulimits` match MinerU's official compose recommendations
- GPU device `0` = GTX 1080 Ti (shares with Whisper, but rarely concurrent)

#### 3.2 Update Makefile

**File**: `onyx_data/deployment/Makefile`

Add these variables and targets:

```makefile
# Add to existing variables
COMPOSE_MINERU_FILE := docker-compose.mineru.yml
COMPOSE_FULL := docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_GPU_FILE) -f $(COMPOSE_MINERU_FILE)

# Build MinerU custom image
build-mineru:
	docker compose -f $(COMPOSE_MINERU_FILE) build mineru
	@echo "MinerU image built successfully"

# Full stack: Onyx + GPU + MinerU
run-full:
	$(COMPOSE_FULL) up -d
	@echo "Onyx + MinerU started."
	@echo "  Onyx UI:   http://localhost:7002"
	@echo "  MinerU UI: http://localhost:7860"

start-full:
	$(COMPOSE_FULL) start

restart-full:
	$(COMPOSE_FULL) restart

logs-mineru:
	$(COMPOSE_FULL) logs -f mineru

shell-mineru:
	$(COMPOSE_FULL) exec mineru bash
```

#### 3.3 Add env vars to `.env`

**File**: `onyx_data/deployment/.env`

Add at the bottom:

```
################################################################################
## MINERU OCR CONFIGURATION
################################################################################
## API key for pushing parsed docs to Onyx (created in Admin > API Keys)
## Must be an Admin key (needed to auto-create the File Connector on first run)
ONYX_API_KEY=
```

Note: `ONYX_CC_PAIR_ID` is **not needed** in `.env`. The `push_to_onyx.py`
script auto-creates a File Connector named `mineru-ocr-documents` on its first
run and caches the `cc_pair_id` in the `mineru_output` Docker volume.

---

### Phase 4: Testing & Verification

#### 4.1 Build the MinerU image

```bash
cd onyx_data/deployment
make build-mineru
```

This will take 10-20 minutes (downloading PyTorch + MinerU models).

#### 4.2 Start the full stack

```bash
make run-full
```

#### 4.3 Verify MinerU is running

1. Open `http://localhost:7860` — should see MinerU Gradio UI
2. Upload a simple digital PDF — verify it parses to Markdown
3. Upload a scanned PDF — verify OCR works
4. Upload a photo of a document — verify it's recognized
5. Upload a DOCX file — verify native parsing

#### 4.4 Verify Onyx integration

1. Open Onyx at `http://localhost:7002`
2. In MinerU Gradio, upload a document with "Push to Onyx" enabled
3. In Onyx, search for content from the uploaded document
4. Verify the document appears in search results
5. Start a chat and ask questions about the document content

#### 4.5 GPU coexistence test

1. Start a Whisper transcription job
2. Simultaneously upload a document to MinerU
3. Both should complete (may be slower due to VRAM sharing)
4. Check `nvidia-smi` to verify VRAM stays within 11 GB total

---

### Phase 5: Connect MCP Clients

Documents pushed to Onyx via MinerU are automatically indexed and available
through Onyx's MCP server. The MCP server is already enabled in `.env`
(`MCP_SERVER_ENABLED=true`) and listens on port **8090**.

#### 5.1 Available MCP tools

| Tool | Description |
|---|---|
| `search_indexed_documents` | Search all indexed docs (including MinerU-ingested). Supports `source_types`, `time_cutoff`, `limit` filters. |
| `search_web` | Search the public internet |
| `open_urls` | Fetch full text content from URLs |

MinerU documents will appear with `source_type: "file"` (because we associate
them with a File Connector). You can filter for them in MCP queries:

```json
{
  "query": "quarterly revenue figures",
  "source_types": ["file"],
  "limit": 10
}
```

#### 5.2 Connect Claude Code

```bash
claude mcp add --transport http onyx http://localhost:8090/ \
  --header "Authorization: Bearer <ONYX_API_KEY>"
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "onyx": {
      "type": "http",
      "url": "http://localhost:8090/",
      "headers": {
        "Authorization": "Bearer <ONYX_API_KEY>"
      }
    }
  }
}
```

#### 5.3 Connect Cursor / Windsurf / other MCP clients

| Setting | Value |
|---|---|
| URL | `http://localhost:8090/` |
| Transport | HTTP (Streamable HTTP) |
| Auth Header | `Authorization: Bearer <ONYX_API_KEY>` |

Refer to each client's MCP configuration docs for exact setup steps.

#### 5.4 Verify MCP can find MinerU documents

After uploading a document through MinerU Gradio:

1. In Claude Code, ask: "Search onyx for [content from your document]"
2. Claude should invoke `search_indexed_documents` and return matching chunks
3. Verify the `source_type` is `"file"` in the results

#### 5.5 Debugging MCP

```bash
# Check MCP server health
curl http://localhost:8090/health

# Use the MCP Inspector for interactive debugging
npx @modelcontextprotocol/inspector
# -> Select Bearer Token auth, paste your API key, connect to http://localhost:8090/
```

---

## Resource Budget

### RAM (15.6 GB total)

| Service | RAM Usage | Notes |
|---|---|---|
| Whisper container | ~2.7 GB | large-v3 model resident |
| MetaTrader container | ~660 MB | |
| Onyx Postgres | ~300 MB | |
| Onyx Vespa | ~1-2 GB | Primary search index |
| Onyx Redis | ~50 MB | |
| Onyx inference_model_server | ~500 MB - 1 GB | Embedding models |
| Onyx indexing_model_server | ~500 MB - 1 GB | |
| Onyx api_server | ~200-500 MB | |
| Onyx background | ~200-500 MB | |
| Onyx web_server (Next.js) | ~200-500 MB | |
| Onyx nginx | ~10 MB | |
| Onyx MinIO | ~100 MB | S3 file store |
| **MinerU** | **~2-3 GB** | **PyTorch + pipeline models (on-demand)** |
| **Total estimated** | **~9-12 GB** | **Leaves ~4-7 GB free** |

With OpenSearch disabled, this should fit. If RAM pressure occurs, MinerU's
models can be configured to load on-demand and unload after idle timeout.

### GPU VRAM

| GPU | Service | VRAM Used | VRAM Total |
|---|---|---|---|
| GPU 0 (1080 Ti) | Whisper (large-v3) | ~2 GB | 11 GB |
| GPU 0 (1080 Ti) | **MinerU pipeline** | **~3-4 GB** | (shared) |
| GPU 1 (1060 3GB) | Onyx embeddings | ~1-2 GB | 3 GB |

Whisper + MinerU on the 1080 Ti: ~5-6 GB total, well within 11 GB.
Concurrent use is possible but not recommended for peak performance.

---

## File Summary

### New files to create

| File | Purpose |
|---|---|
| `onyx_data/deployment/mineru/Dockerfile` | Custom MinerU image for Pascal GPUs |
| `onyx_data/deployment/mineru/push_to_onyx.py` | Onyx Ingestion API client module |
| `onyx_data/deployment/mineru/mineru_with_onyx.py` | Custom Gradio app wrapping MinerU + Onyx push |
| `onyx_data/deployment/docker-compose.mineru.yml` | Docker compose overlay adding MinerU sidecar |

### Files to modify

| File | Change |
|---|---|
| `onyx_data/deployment/.env` | Add `ONYX_API_KEY`, set `OPENSEARCH_FOR_ONYX_ENABLED=false` |
| `onyx_data/deployment/Makefile` | Add `build-mineru`, `run-full`, `start-full`, `logs-mineru` targets |

### No changes needed

| File | Reason |
|---|---|
| `docker-compose.yml` | Onyx base config unchanged |
| `docker-compose.gpu.yml` | GPU config unchanged |
| Whisper/transcription files | Completely separate |

---

## Onyx Ingestion API Reference

### Endpoint

```
POST /onyx-api/ingestion
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

### Minimum payload

```json
{
  "document": {
    "semantic_identifier": "Invoice-2024-001.pdf",
    "sections": [
      { "text": "Parsed content from page 1..." },
      { "text": "Parsed content from page 2..." }
    ],
    "source": "file",
    "metadata": {
      "parsed_by": "mineru",
      "file_type": ".pdf"
    },
    "from_ingestion_api": true
  },
  "cc_pair_id": 1
}
```

Note: `cc_pair_id` is auto-determined by `push_to_onyx.py`. Using
`"source": "file"` (instead of `"ingestion_api"`) ensures documents appear as
`"file"` type in MCP's `source_types` filter and in the Onyx Admin > Connectors
page.

### File Connector auto-creation APIs

```bash
# Step 1: Create connector (one-time, done automatically by push_to_onyx.py)
POST /manage/admin/connector
{
  "name": "mineru-ocr-documents",
  "source": "file",
  "input_type": "poll",
  "access_type": "PUBLIC",
  "connector_specific_config": {}
}
# Returns: {"id": <connector_id>, ...}

# Step 2: Associate with empty credential (File connectors don't need auth)
PUT /manage/admin/connector/<connector_id>/credential/0
# Returns cc_pair_id in response
```

### Response

```json
{
  "document_id": "ingestion_api__abcdef123",
  "already_existed": false
}
```

### Full payload fields

| Field | Required | Description |
|---|---|---|
| `document.semantic_identifier` | Yes | Display name in Onyx UI |
| `document.sections[].text` | Yes | Content text (one per page/section) |
| `document.sections[].link` | No | Optional URL link for the section |
| `document.source` | No | One of the `DocumentSource` enum values |
| `document.metadata` | Yes | Arbitrary key-value pairs, used as tags/filters |
| `document.id` | No | Custom ID; auto-generated if omitted |
| `document.doc_updated_at` | No | ISO 8601 datetime |
| `document.from_ingestion_api` | No | Set `true` to mark provenance |
| `cc_pair_id` | No | Associates doc with a File Connector |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PyTorch 2.6 + CUDA 11.8 incompatible with MinerU pipeline models | Low | High | Test during build. Fall back to `torch 2.4.0` or `cuda 12.1` if needed. |
| MinerU Gradio app not easily extensible for Onyx hook | Medium | Medium | Fall back to output directory watcher (Approach B). |
| RAM exhaustion with all services running | Medium | High | Disable OpenSearch. Monitor with `docker stats`. Stop Whisper when not needed. |
| Whisper + MinerU concurrent GPU OOM on 1080 Ti | Low | Medium | Document that simultaneous use is not recommended. Add VRAM monitoring. |
| Onyx Ingestion API changes across versions | Low | Low | Pin Onyx version in `.env` (`IMAGE_TAG=edge`). |
| MinerU model download fails during Docker build | Low | Medium | Use `mineru_model_cache` volume to persist across rebuilds. |

---

## Future Enhancements (Out of Scope)

- **Batch processing**: Folder watcher for automated ingestion of document directories
- **Onyx custom connector**: Build a proper MinerU connector plugin for Onyx
- **OCRmyPDF integration**: Produce searchable PDFs alongside Onyx ingestion
- **MinerU VLM engine**: Upgrade GPUs to Ampere+ to unlock VLM backend (95+ accuracy)
- **Nginx proxy**: Route MinerU through Onyx's nginx at `/mineru/` path
- **Progress notifications**: WebSocket updates from MinerU parse jobs

---

## Quick Reference

### URLs (after deployment)

| Service | URL |
|---|---|
| Onyx UI | http://localhost:7002 |
| Onyx API docs | http://localhost:7002/api/docs |
| Onyx MCP server | http://localhost:8090/ |
| MinerU Gradio UI | http://localhost:7860 |

### Key commands

```bash
cd onyx_data/deployment

# Build MinerU image (first time, ~15 min)
make build-mineru

# Start everything (Onyx + GPU + MinerU)
make run-full

# Start just Onyx with GPU (no MinerU)
make run-gpu

# View MinerU logs
make logs-mineru

# Shell into MinerU container
make shell-mineru

# Check resource usage
docker stats --no-stream

# Check GPU usage
nvidia-smi
```
