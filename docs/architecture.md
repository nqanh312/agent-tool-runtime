# Architecture and design notes

Agent Tool Runtime separates model interaction, policy enforcement, and external-service access so each layer can be inspected and tested independently.

## Request flow

For each user message, the FCI planner returns a structured route and any explicit facts or preferences found in the message. Depending on that route, the runtime can recall memory, browse Drive, read a Drive file, save the current artifact, or continue as general chat.

Planner failures are fail-closed: the chat model remains available, but the turn receives no Drive or memory tools. This prevents a planner outage from silently broadening the model's capabilities.

Every model-requested tool call enters the central registry. The registry validates its input schema, authenticates the principal, checks required permissions, applies an in-process rate limit, executes the handler, and records the outcome in the audit trail.

## Trust boundaries

Retrieved memories and Drive documents are treated as untrusted data rather than instructions. User identity and permissions are supplied by the application, not by the model.

Memory writes are also separated from the chat model:

- Structured facts and preferences are written through an internal `upsert_user_memory` operation.
- The current document is saved from trusted server-side artifact state through `save_current_document`.
- These operations use the registry's authentication, authorization, rate-limiting, and audit controls but are not exposed as chat-model tools.

Google login and Drive consent are separate OAuth transactions. Offline Drive grants are encrypted and owned by the stable application user ID; accounts are not merged by email address.

## Storage

PostgreSQL stores users, role assignments, refresh sessions, security events, conversations, messages, current artifacts, and tool audit logs. The application uses Alembic migrations and does not create or alter tables automatically at startup.

Qdrant stores user-partitioned long-term memory. Documents are converted to Markdown blocks, packed into token-limited chunks with overlap, and embedded with the configured OpenAI or FCI embedding model. Retrieval combines dense-vector and indexed BM25 sparse searches with reciprocal-rank fusion.

Use a new Qdrant collection when changing the embedding model or dimension. `scripts/migrate_qdrant_hybrid.py` upgrades collections created by older versions to the current sparse-vector and payload-index schema.

## Process-local state

Login and tool-call rate limits, active agent sessions, and conversation locks are held in the application process. The bounded TTL/LRU behavior is suitable for the current single-process deployment, but a multi-worker deployment requires shared coordination such as Redis or database-backed state.

## Benchmarking

The vector-store benchmark generates isolated test data and reports p50, p95, p99, and queries per second as JSON lines:

```bash
python -m scripts.benchmark_vectorstore --sizes 1000,10000,100000 --samples 200
```

It creates a uniquely named temporary collection and removes only that collection when complete. Pass `--keep` to retain it for inspection.
