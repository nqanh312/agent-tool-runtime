# Agent Tool Runtime

> A modular runtime for building auditable AI agents with secure tool execution and long-term memory.

Agent Tool Runtime is an experimental Python framework for connecting Anthropic, OpenAI, or FCI/FPT Cloud models to external tools through a governed execution pipeline. It combines a central tool registry, scoped access control, audit logging, document ingestion, and RAG-based memory in a compact codebase designed for learning and research.

The project explores a practical question: **how can an AI agent use external capabilities while keeping tool execution explicit, inspectable, and controllable?**

> [!IMPORTANT]
> This project is under active development. The agent loop, interfaces, governed tool execution, Google Drive tools, document conversion, and hybrid RAG memory are available. See [Development status](#development-status) before running the project.

## Highlights

- Provider-independent agent loop with Anthropic and OpenAI-compatible tool use.
- Runtime provider selection for Anthropic, OpenAI, and FCI/FPT Cloud.
- Central registry for tool definitions and invocation.
- Six-stage execution pipeline for validation, authentication, authorization, rate limiting, execution, and auditing.
- Read-only Google Drive integration through a service account.
- Document conversion for PDF, DOCX, XLS/XLSX, PPTX, HTML, and text formats using MarkItDown.
- FCI-based extraction of explicit user facts and preferences into structured current-state memory before the chat model runs.
- FCI turn routing that selects RAG, Drive browsing, Drive reading, artifact saving, or general chat in the same classification call.
- Sanitized Markdown rendering for headings, tables, lists, code blocks, and links in web chat.
- Semantic long-term memory using OpenAI or FCI embeddings and Qdrant.
- CLI and FastAPI interfaces with session-isolated conversation history.

## Architecture

![Agent Tool Runtime architecture](./agent_tool_flow.png)

Every tool call is designed to pass through the same policy boundary:

```text
User request
    -> Configured model provider
    -> Tool registry
       -> Validate schema
       -> Authenticate caller
       -> Check scopes
       -> Enforce rate limit
       -> Execute handler
       -> Write audit log
    -> Tool result
    -> Claude response
```

This separation keeps model reasoning, policy enforcement, and external service access independent and easier to test.

## Project structure

```text
.
|-- agent.py                 # Provider-independent conversation and tool loop
|-- config.py                # Environment-based configuration
|-- main.py                  # Interactive CLI
|-- server.py                # FastAPI service and web UI
|-- registry/
|   |-- models.py            # Tool metadata model
|   `-- registry.py          # Policy and execution pipeline
|-- services/
|   |-- drive_service.py     # Google Drive API adapter
|   |-- embedding.py         # OpenAI embedding adapter
|   |-- file_reader.py       # Document-to-Markdown conversion
|   |-- llm.py               # Anthropic/OpenAI/FCI model adapters
|   `-- vectorstore.py       # Qdrant memory adapter
|-- tools/
|   |-- google_drive.py      # Google Drive tools
|   |-- memory.py            # Long-term memory tools
|   `-- read_file.py         # Local file-reading tool
|-- static/
|   `-- index.html           # Browser-based chat interface
|-- tests/                   # Automated tests
`-- requirements.txt
```

## Requirements

- Python 3.10 or later
- An API key for the selected model provider (Anthropic, OpenAI, or FCI)
- An API key authorized for the selected OpenAI or FCI embedding model
- Docker Desktop or an accessible Qdrant instance
- A Google Cloud service account for Google Drive integration

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/agent-tool-runtime.git
cd agent-tool-runtime
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env`, then configure the selected provider:

```env
LLM_PROVIDER=anthropic
LLM_MODEL=

ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-sonnet-4-20250514

OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
MEMORY_CHUNK_TOKENS=500
MEMORY_CHUNK_OVERLAP_TOKENS=75

FCI_API_KEY=your_fci_api_key
FCI_MODEL=gemma-4-31B-it
FCI_BASE_URL=https://mkp-api.fptcloud.com/v1
MEMORY_EXTRACTION_MODEL=gemma-4-31B-it
TURN_PLANNER_MODEL=gemma-4-31B-it
MEMORY_EXTRACTION_MIN_CONFIDENCE=0.7
MEMORY_RELEVANCE_MIN_SEMANTIC_SCORE=0.7

QDRANT_HOST=localhost
QDRANT_PORT=6333
MEMORY_COLLECTION=agent_memory

GOOGLE_SERVICE_ACCOUNT_FILE=credentials.json
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id
```

`GOOGLE_DRIVE_FOLDER_ID` is optional. Leave it empty to list every file accessible to the service account.

For Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY` instead of the local host/port. Memory is partitioned by the authenticated registry user. Facts and preferences up to the configured token limit remain intact. Documents are parsed into Markdown headings, paragraphs, lists, tables, and fenced code blocks; blocks are packed into token-limited chunks with overlap, and only oversized blocks are hard-split. Retrieval combines cosine similarity with BM25 ranking.

Select exactly one chat-model provider with `LLM_PROVIDER`:

| Provider | Value | Required key | Default model |
| --- | --- | --- | --- |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `OPENAI_LLM_MODEL` |
| FCI/FPT Cloud | `fci` | `FCI_API_KEY` | `FCI_MODEL` |

`LLM_MODEL` is an optional global override. Leave it empty to use the model configured for the selected provider. FCI uses its OpenAI-compatible Chat Completions endpoint; `FPT_API_KEY` is also accepted as an alias for `FCI_API_KEY`.

Select `EMBEDDING_PROVIDER=openai` or `EMBEDDING_PROVIDER=fci` independently of the chat provider. For FCI embeddings, the API key must be authorized for an embedding model. Example:

```env
EMBEDDING_PROVIDER=fci
EMBEDDING_MODEL=multilingual-e5-large
EMBEDDING_DIM=1024
MEMORY_COLLECTION=agent_memory_fci_1024
```

Use a new collection whenever the embedding dimension or embedding model changes; existing vectors from another model are not compatible.

Turn planning and automatic fact extraction use one FCI request, independently of
`LLM_PROVIDER`. The planner makes one forced structured-output call for each user
message. It routes the turn to RAG recall, Drive browsing, Drive file reading,
current-artifact saving, or general chat. It also assigns
`category`, an English `snake_case` topic, normalized value, polarity, and confidence;
compound statements are split into separate memories. A preference is keyed by
`category + topic + canonical_value`, so a later polarity change updates the same
record instead of leaving contradictory active records. Set
`TURN_PLANNER_MODEL` to an FCI chat model available to your API key, and adjust
`MEMORY_EXTRACTION_MIN_CONFIDENCE` to control which extracted items are persisted.

Memory writes have separate ownership boundaries. FCI extraction invokes the internal
`upsert_user_memory` operation for structured facts and preferences. Current files are
saved through `save_current_document`, which reads trusted server-side artifact state
instead of asking the model to reproduce the content. The underlying write operations
are registered for authentication, authorization, rate limiting, and auditing but are
never exposed to the chat model. Both paths share the same chunking, embedding, and
Qdrant persistence implementation.

> [!WARNING]
> Never commit `.env`, API keys, or service-account credentials. The default `.gitignore` excludes these files.

### 4. Start Qdrant

```bash
docker run -d --name agent-tool-runtime-qdrant \
  --restart unless-stopped \
  -p 127.0.0.1:6333:6333 \
  -v agent_tool_runtime_qdrant:/qdrant/storage \
  qdrant/qdrant
```

The Qdrant dashboard will be available at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

Useful container commands:

```bash
docker stop agent-tool-runtime-qdrant
docker start agent-tool-runtime-qdrant
docker logs agent-tool-runtime-qdrant
```

### 5. Configure Google Drive (optional)

1. Create a project in Google Cloud Console.
2. Enable the Google Drive API.
3. Create a service account and download its JSON key.
4. Set the key path with `GOOGLE_SERVICE_ACCOUNT_FILE`.
5. Share the target Drive folder with the service account email.
6. Set `GOOGLE_DRIVE_FOLDER_ID` if access should default to one folder.

The integration requests only the `https://www.googleapis.com/auth/drive.readonly` scope.

## Usage

### Command-line interface

```bash
python main.py
```

| Command | Description |
| --- | --- |
| `/help` | Show available commands |
| `/clear` | Clear the current conversation |
| `/audit` | Display tool-call audit entries |
| `/memory` | List stored memories |
| `/quit` | Exit the application |

### Web interface

```bash
python server.py
```

Open [http://localhost:9004](http://localhost:9004). The health endpoint is available at [http://localhost:9004/api/health](http://localhost:9004/api/health).

## HTTP API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/chat` | Send a message to an agent session |
| `POST` | `/api/clear` | Clear a session's conversation history |
| `GET` | `/api/audit?session_id=...` | Retrieve tool-call audit entries |
| `GET` | `/api/memories?session_id=...` | List the user's current facts and preferences |
| `GET` | `/api/documents?session_id=...` | List saved document sources and chunk counts |
| `GET` | `/api/health` | Check service availability |

Example request:

```bash
curl -X POST http://localhost:9004/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"List my Drive files."}'
```

## Development status

| Component | Status |
| --- | --- |
| Provider-independent conversation loop | Available |
| Anthropic adapter | Available |
| OpenAI-compatible adapter (OpenAI and FCI) | Available |
| CLI and FastAPI interfaces | Available |
| Tool registration | Available |
| Google Drive listing, download, and reading | Available |
| Schema validation | Available |
| API-key authentication | Available (demo only) |
| Scope authorization | Available |
| Sliding-window rate limiter | Available (in-memory) |
| Tool execution and audit logging | Available (in-memory) |
| MarkItDown conversion | Available |
| OpenAI and FCI embeddings | Available |
| FCI structured fact/preference extraction | Available |
| Qdrant hybrid memory (vector + BM25) | Available |

The in-memory authentication, rate limiting, and audit storage are intended for demonstration rather than production use. Long-term memory itself is persisted in Qdrant and remains available after conversation history is cleared or the browser is reloaded.

## Research directions

- Policy-aware and capability-based tool access.
- Sandboxed execution for untrusted tools.
- Durable memory with retrieval-quality evaluation.
- Multi-agent orchestration and delegation.
- Model fallback, routing, and provider health checks.
- Tool-selection, latency, reliability, and safety benchmarks.

## Testing

Run the test suite from the project root:

```bash
python -m unittest discover -v
```

## Security notes

- Do not hard-code or commit credentials.
- The in-memory users and service API keys are development fixtures, not a production identity system.
- CORS currently permits all origins and must be restricted before deployment.
- Keep local Qdrant bound to `127.0.0.1`; use authentication and network controls when exposing it remotely.
- Local file access should be sandboxed or allowlisted before use in a multi-user environment.
