# Agent Tool Runtime

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

> A modular Python runtime for auditable AI agents, governed tool execution, and long-term memory.

Agent Tool Runtime connects Anthropic, OpenAI, or FCI/FPT Cloud models to external tools through a policy-controlled execution pipeline. It includes user authentication, scoped permissions, audit logs, Google Drive ingestion, hybrid RAG memory, a CLI, and a FastAPI web application.

> [!IMPORTANT]
> This is an experimental learning under active development. Review [Current limitations](#current-limitations) before deploying it beyond a trusted environment.

## Highlights

- Provider-independent agent loop for Anthropic and OpenAI-compatible APIs.
- Central tool registry with schema validation, authentication, authorization, rate limiting, execution, and auditing.
- Per-user Google Drive access through OAuth 2.0.
- PDF, DOCX, XLS/XLSX, PPTX, HTML, and text conversion with MarkItDown.
- Structured fact and preference extraction with FCI turn planning.
- User-isolated hybrid retrieval using dense vectors, BM25 sparse vectors, and Qdrant.
- PostgreSQL-backed authentication, conversations, artifacts, and audit history.
- CLI and responsive browser-based chat interfaces.

## Architecture

![Agent Tool Runtime architecture](./agent_tool_flow.png)

Every tool operation crosses the same policy boundary:

```text
User request
    -> FCI turn planner
    -> Agent loop
       -> Pre-routed operation or model tool request
       -> Tool registry
          -> Validate schema
          -> Authenticate caller
          -> Check permissions
          -> Enforce rate limit
          -> Execute handler
          -> Write audit log
       -> Tool result
    -> Configured chat model
    -> Model response
```

See [Architecture and design notes](./docs/architecture.md) for the memory model, routing behavior, and trust boundaries.

## Requirements

- Python 3.10 or later
- Docker Desktop, or accessible PostgreSQL and Qdrant instances
- An API key for the selected chat provider
- An API key for the selected embedding provider
- Optional: an FCI API key for turn planning, tool routing, and automatic memory extraction
- Optional: a Google Cloud Web OAuth client for Google login and Drive access

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/nqanh312/agent-tool-runtime.git
cd agent-tool-runtime
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure the environment

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux, use `cp .env.example .env` instead. At minimum, review these settings:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

DATABASE_URL=postgresql+psycopg://agent:agent@localhost:5432/agent_db
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

JWT_SECRET=replace_with_at_least_32_random_characters
```

Generate a suitable local JWT secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Chat and embedding providers are selected independently:

| Purpose | Provider value | API key | Default model |
| --- | --- | --- | --- |
| Chat | `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| Chat | `openai` | `OPENAI_API_KEY` | `OPENAI_LLM_MODEL` |
| Chat | `fci` | `FCI_API_KEY` | `FCI_MODEL` |
| Embeddings | `openai` | `OPENAI_API_KEY` | `EMBEDDING_MODEL` |
| Embeddings | `fci` | `FCI_API_KEY` | `EMBEDDING_MODEL` |

`LLM_MODEL` can override the model for the selected chat provider. Use a new `MEMORY_COLLECTION` whenever the embedding model or dimension changes.

Turn planning always uses FCI, independently of `LLM_PROVIDER`. Without `FCI_API_KEY`, ordinary chat remains available, but the planner fails closed to general chat and does not route the turn to Drive or memory tools.

The complete set of options and documented defaults is in [`.env.example`](./.env.example).

> [!WARNING]
> Never commit `.env`, OAuth secrets, encryption keys, or refresh tokens. Local secret files are excluded by `.gitignore`.

### 3. Start the data services

```bash
docker compose up -d
alembic upgrade head
python -m scripts.bootstrap_admin --username admin
```

The bootstrap command securely prompts for the initial administrator password. PostgreSQL and Qdrant persist their data in named Docker volumes.

The Qdrant dashboard is available at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

If upgrading a collection created by an older version of this project, migrate it with:

```bash
python -m scripts.migrate_qdrant_hybrid --dry-run
python -m scripts.migrate_qdrant_hybrid
```

## Usage

### Web interface

```bash
python server.py
```

Open [http://localhost:9004](http://localhost:9004). API documentation is available at [http://localhost:9004/docs](http://localhost:9004/docs), and the health endpoint is at [http://localhost:9004/api/health](http://localhost:9004/api/health).

### Command-line interface

Set `CLI_USER_ID` to an existing application user, then run:

```bash
python main.py
```

`CLI_USER_ID` selects the local process identity and permissions; it is not an authentication credential. Run the CLI only on a trusted machine.

| Command | Description |
| --- | --- |
| `/help` | Show available commands |
| `/clear` | Clear the current conversation |
| `/audit` | Display tool-call audit entries |
| `/memory` | List stored memories |
| `/quit` | Exit the application |

## Google Drive integration

Google integration is optional. To enable it:

1. Enable the Google Drive API in a Google Cloud project.
2. Configure the OAuth consent screen.
3. Create an OAuth client of type **Web application**.
4. Add `http://localhost:9004/api/auth/google/callback` as an authorized redirect URI.
5. Configure the `GOOGLE_OAUTH_*` variables in `.env`.
6. Generate `GOOGLE_TOKEN_ENCRYPTION_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

The current list/search-all-Drive flow uses the restricted `drive.readonly` scope. A public Google integration should prefer `drive.file` with Google Picker and complete any required Google verification.

## HTTP API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/auth/login` | Authenticate a local user |
| `POST` | `/api/auth/complete-password-change` | Complete a required first password change |
| `POST` | `/api/auth/refresh` | Rotate the refresh token |
| `POST` | `/api/auth/logout` | Revoke the current refresh session |
| `GET` | `/api/auth/me` | Return the authenticated user |
| `POST` | `/api/auth/change-password` | Change the current user's password |
| `GET` | `/api/auth/google/start` | Start Google login |
| `GET` | `/api/auth/google/callback` | Complete Google OAuth |
| `POST` | `/api/integrations/google-drive/authorize` | Start per-user Drive consent |
| `GET` | `/api/integrations/google-drive/status` | Read Drive connection state |
| `POST` | `/api/integrations/google-drive/disconnect` | Revoke and delete a Drive grant |
| `GET`, `POST` | `/api/admin/users` | List or create users |
| `PATCH` | `/api/admin/users/{user_id}` | Update a user |
| `POST` | `/api/admin/users/{user_id}/reset-password` | Reset a user's password |
| `POST` | `/api/chat` | Send a chat message |
| `GET` | `/api/conversations` | List the current user's conversations |
| `GET` | `/api/conversations/{id}/messages` | Load stored messages |
| `GET` | `/api/audit` | Retrieve tool-call audit entries |
| `GET` | `/api/memories` | List current facts and preferences |
| `GET` | `/api/documents` | List saved document sources |
| `GET` | `/api/health` | Check service health |

Protected routes use an access token returned by `/api/auth/login`:

```bash
curl -X POST http://localhost:9004/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your-password"}'
```

## Project structure

```text
.
|-- agent.py                 # Conversation, routing, and tool loop
|-- config.py                # Environment-based configuration
|-- main.py                  # Interactive CLI
|-- server.py                # FastAPI service and web UI
|-- registry/                # Tool metadata and policy pipeline
|-- services/                # LLM, storage, OAuth, auth, and rendering adapters
|-- tools/                   # Google Drive and memory tools
|-- migrations/              # Alembic database migrations
|-- scripts/                 # Administration, migration, and benchmark utilities
|-- static/                  # Browser-based chat interface
|-- tests/                   # Automated tests
`-- docs/                    # Design and operational documentation
```

## Testing

Run the unit test suite from the project root:

```bash
python -m unittest discover -v
```

The optional PostgreSQL integration test requires running PostgreSQL and migrated tables:

```bash
RUN_POSTGRES_TESTS=1 python -m unittest tests.test_postgres_integration -v
```

In Windows PowerShell, set the variable with `$env:RUN_POSTGRES_TESTS = "1"` before running the command.

## Current limitations

- Rate-limit buckets, active agent sessions, and conversation locks are process-local.
- Tool routing and automatic memory extraction currently depend on the FCI planner.
- The bundled deployment configuration targets a single trusted application instance.
- Google Drive list/search uses a restricted OAuth scope that may require verification for public use.
- Sandboxed execution for untrusted third-party tools is not yet implemented.

## Roadmap

- Shared rate limiting and session coordination for multi-worker deployments.
- Capability-based access policies and sandboxed tools.
- Retrieval-quality, tool-selection, latency, and safety evaluations.
- Provider health checks, routing, and fallback strategies.

## Security

Use high-entropy secrets, HTTPS secure cookies, exact CORS origins, and authenticated private data services in production. See [SECURITY.md](./SECURITY.md) for the reporting process and deployment considerations.

## License

This project is licensed under the [MIT License](./LICENSE).
