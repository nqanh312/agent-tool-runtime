"""Tests for durable, user-scoped hybrid memory."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient

import agent as agent_module
from agent import Agent
from config import EMBEDDING_DIM
from registry import ToolRegistry
from registry.registry import AUDIT_LOG, rate_limiter
from services import embedding, memory_extractor, vectorstore
from services.llm import ModelResponse
from services.memory_extractor import (
    ExtractedMemory,
    TurnPlan,
    plan_user_turn,
)
from scripts import migrate_qdrant_hybrid
from tools import memory

ADMIN = {
    "user_id": "user_admin", "role": "admin", "is_active": True,
    "permissions": ["drive:read", "memory:read", "memory:write"],
}
STANDARD_USER = {
    "user_id": "user_standard", "role": "user", "is_active": True,
    "permissions": ["drive:read", "memory:read", "memory:write"],
}
GUEST = {
    "user_id": "user_guest", "role": "guest", "is_active": True,
    "permissions": ["drive:read", "memory:read"],
}


def _vector(first: float = 1.0, second: float = 0.0) -> list[float]:
    return [first, second] + [0.0] * (EMBEDDING_DIM - 2)


class _FinalResponseLLM:
    provider = "test"
    model = "test-model"

    def __init__(self):
        self.requests = []

    def complete(self, **_request):
        self.requests.append(_request)
        return ModelResponse(text="Đã hiểu.", stop_reason="stop")


class _SequenceResponseLLM:
    provider = "test"
    model = "test-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


class MemoryExtractorTests(unittest.TestCase):
    def setUp(self):
        AUDIT_LOG.clear()
        rate_limiter._buckets.clear()

    @staticmethod
    def _client_returning(
        items,
        intent="general_chat",
        search_query="",
        file_query="",
        drive_fallback=False,
    ):
        arguments = json.dumps(
            {
                "intent": intent,
                "search_query": search_query,
                "file_query": file_query,
                "drive_fallback": drive_fallback,
                "memories": items,
            }
        )
        tool_call = SimpleNamespace(
            function=SimpleNamespace(arguments=arguments),
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(tool_calls=[tool_call]),
                )
            ]
        )
        create = unittest.mock.Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client, create

    def test_fci_splits_and_classifies_compound_preferences(self):
        client, create = self._client_returning(
            [
                {
                    "category": "user_preference",
                    "topic": "Programming Language",
                    "value": "Java",
                    "canonical_value": "java",
                    "polarity": "like",
                    "memory_text": "The user likes Java.",
                    "confidence": 0.98,
                },
                {
                    "category": "user_preference",
                    "topic": "programming-language",
                    "value": "Python",
                    "canonical_value": "Python",
                    "polarity": "dislike",
                    "memory_text": "The user does not like Python.",
                    "confidence": 0.99,
                },
            ]
        )
        with patch.object(memory_extractor, "_get_client", return_value=client):
            plan = plan_user_turn(
                "I like Java, but I do not like Python."
            )

        memories = list(plan.memories)
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0].topic, "programming_language")
        self.assertEqual(memories[1].topic, "programming_language")
        self.assertEqual(memories[1].canonical_value, "python")
        self.assertEqual(memories[1].polarity, "dislike")
        request = create.call_args.kwargs
        self.assertEqual(request["tool_choice"]["function"]["name"], "plan_turn")
        self.assertEqual(request["temperature"], 0)

    def test_fci_can_return_no_memory_for_a_question(self):
        client, _create = self._client_returning(
            [],
            intent="recall_memory",
            search_query="user preferences",
        )
        with patch.object(memory_extractor, "_get_client", return_value=client):
            plan = plan_user_turn("What do I like?")

        self.assertEqual(plan.intent, "recall_memory")
        self.assertEqual(plan.search_query, "user preferences")
        self.assertEqual(list(plan.memories), [])

    def test_fci_routes_local_file_reads_separately(self):
        client, _create = self._client_returning(
            [],
            intent="read_local_file",
            file_query=r"C:\docs\notes.pdf",
        )
        with patch.object(memory_extractor, "_get_client", return_value=client):
            plan = plan_user_turn(r"Read C:\docs\notes.pdf")

        self.assertEqual(plan.intent, "read_local_file")
        self.assertEqual(plan.file_query, r"C:\docs\notes.pdf")

    def test_agent_saves_extracted_memory_before_calling_the_llm(self):
        captured = []

        def fake_save(text, _vector_value, metadata):
            captured.append((text, metadata))
            return {"id": "memory-1"}

        with (
            patch.object(
                agent_module,
                "plan_user_turn",
                return_value=TurnPlan(
                    intent="general_chat",
                    memories=(ExtractedMemory(
                        content="The user likes Python.",
                        category="user_preference",
                        topic="programming_language",
                        value="Python",
                        canonical_value="python",
                        polarity="like",
                        confidence=0.99,
                    ),),
                ),
            ),
            patch.object(memory.embedding, "embed_texts", return_value=[_vector()]),
            patch.object(memory.vectorstore, "save_memory", side_effect=fake_save),
        ):
            llm = _FinalResponseLLM()
            response = Agent(principal=ADMIN, llm_client=llm).run("I like Python")

        self.assertEqual(response, "Đã hiểu.")
        self.assertEqual(captured[0][0], "The user likes Python.")
        self.assertEqual(captured[0][1]["category"], "user_preference")
        self.assertEqual(captured[0][1]["topic"], "programming_language")
        self.assertEqual(captured[0][1]["polarity"], "like")
        self.assertEqual(AUDIT_LOG[-1]["tool"], "upsert_user_memory")
        self.assertNotIn(
            "upsert_user_memory",
            {tool["name"] for tool in llm.requests[0]["tools"]},
        )
        self.assertNotIn(
            "save_document_memory",
            {tool["name"] for tool in llm.requests[0]["tools"]},
        )

    def test_guest_does_not_run_automatic_memory_writes(self):
        extracted = ExtractedMemory(
            content="The user likes Python.", category="user_preference",
            topic="programming_language", value="Python",
            canonical_value="python", polarity="like", confidence=0.99,
        )
        with patch.object(
            agent_module, "plan_user_turn",
            return_value=TurnPlan(intent="general_chat", memories=(extracted,)),
        ):
            response = Agent(principal=GUEST, llm_client=_FinalResponseLLM()).run(
                "I like Python"
            )
        self.assertEqual(response, "Đã hiểu.")
        self.assertEqual(AUDIT_LOG, [])

    def test_agent_can_upsert_fact_and_save_document_in_the_same_turn(self):
        extracted = ExtractedMemory(
            content="The user likes Python.",
            category="user_preference",
            topic="programming_language",
            value="Python",
            canonical_value="python",
            polarity="like",
            confidence=0.99,
        )
        with (
            patch.object(
                agent_module,
                "plan_user_turn",
                return_value=TurnPlan(
                    intent="save_current_document",
                    memories=(extracted,),
                ),
            ),
            patch.object(memory.embedding, "embed_texts", return_value=[_vector()]),
            patch.object(
                memory.vectorstore,
                "save_memory",
                return_value={"id": "preference-1"},
            ),
            patch.object(
                agent_module,
                "save_document_memory",
                return_value={"status": "saved", "chunks_saved": 1},
            ) as document_save,
        ):
            llm = _FinalResponseLLM()
            tested_agent = Agent(principal=ADMIN, llm_client=llm)
            tested_agent.last_artifact = {
                "file_id": "file-1",
                "file_name": "assignment.pptx",
                "mime_type": "application/presentation",
                "content": "# Displayed file\n\nDocument body.",
                "content_hash": "abc123",
                "truncated": False,
                "total_characters": 32,
            }
            response = tested_agent.run("I like Python. Save the displayed file too.")

        self.assertEqual(response, "Đã hiểu.")
        self.assertEqual(
            [entry["tool"] for entry in AUDIT_LOG],
            ["upsert_user_memory", "save_current_document"],
        )
        document_save.assert_called_once()
        self.assertEqual(document_save.call_args.kwargs["file_id"], "file-1")
        self.assertEqual(
            document_save.call_args.kwargs["file_name"],
            "assignment.pptx",
        )
        offered_tools = {tool["name"] for tool in llm.requests[0]["tools"]}
        self.assertNotIn("upsert_user_memory", offered_tools)
        self.assertNotIn("save_document_memory", offered_tools)

    def test_general_route_does_not_expose_memory_or_drive_tools(self):
        llm = _FinalResponseLLM()
        with patch.object(
            agent_module,
            "plan_user_turn",
            return_value=TurnPlan(intent="general_chat"),
        ):
            response = Agent(principal=ADMIN, llm_client=llm).run("Hello")

        self.assertEqual(response, "Đã hiểu.")
        tool_names = {tool["name"] for tool in llm.requests[0]["tools"]}
        self.assertNotIn("save_document_memory", tool_names)
        self.assertNotIn("upsert_user_memory", tool_names)
        self.assertNotIn("list_drive_files", tool_names)
        self.assertNotIn("read_file", tool_names)

    def test_local_file_route_only_exposes_read_file(self):
        llm = _FinalResponseLLM()
        with patch.object(
            agent_module,
            "plan_user_turn",
            return_value=TurnPlan(
                intent="read_local_file",
                file_query=r"C:\docs\notes.pdf",
            ),
        ):
            Agent(principal=ADMIN, llm_client=llm).run(r"Read C:\docs\notes.pdf")

        self.assertEqual(
            {tool["name"] for tool in llm.requests[0]["tools"]},
            {"read_file"},
        )

    def test_agent_retries_pseudo_tool_text_and_returns_clean_answer(self):
        pseudo_call = json.dumps(
            {
                "action": "dalle.text2im",
                "action_input": '{"prompt":"a cat"}',
                "thought": "I will call an image tool.",
            }
        )
        llm = _SequenceResponseLLM(
            [
                ModelResponse(text=pseudo_call, stop_reason="stop"),
                ModelResponse(
                    text="I cannot generate images in this session.",
                    stop_reason="stop",
                ),
            ]
        )
        with patch.object(
            agent_module,
            "plan_user_turn",
            return_value=TurnPlan(intent="general_chat"),
        ):
            tested_agent = Agent(principal=ADMIN, llm_client=llm)
            response = tested_agent.run("Draw a cat")

        self.assertEqual(response, "I cannot generate images in this session.")
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(llm.requests[0]["tools"], [])
        self.assertIn(
            "previous response was rejected",
            llm.requests[1]["system_prompt"],
        )
        self.assertNotIn(pseudo_call, str(tested_agent.conversation_history))

    def test_agent_falls_back_after_repeated_pseudo_tool_text(self):
        pseudo_call = json.dumps(
            {
                "action": "dalle.text2im",
                "action_input": '{"prompt":"a cat"}',
                "thought": "call tool",
            }
        )
        llm = _SequenceResponseLLM(
            [
                ModelResponse(text=pseudo_call, stop_reason="stop"),
                ModelResponse(
                    text=f"```json\n{pseudo_call}\n```",
                    stop_reason="stop",
                ),
            ]
        )
        with patch.object(
            agent_module,
            "plan_user_turn",
            return_value=TurnPlan(intent="general_chat"),
        ):
            response = Agent(principal=ADMIN, llm_client=llm).run("Vẽ giúp tôi ảnh con mèo")

        self.assertEqual(len(llm.requests), 2)
        self.assertIn("chưa được cấu hình công cụ tạo ảnh", response)
        self.assertNotIn("action", response)
        self.assertNotIn("thought", response)

    def test_recall_route_prefetches_rag_and_does_not_expose_drive(self):
        llm = _FinalResponseLLM()
        stored = [
            {
                "id": "document-1",
                "text": "The assignment is to build a Drive Agent.",
                "metadata": {"category": "document", "user_id": "user_admin"},
                "score": 0.03,
                "semantic_score": 0.85,
                "lexical_score": 1.2,
            }
        ]
        with (
            patch.object(
                agent_module,
                "plan_user_turn",
                return_value=TurnPlan(
                    intent="recall_memory",
                    search_query="Drive Agent assignment requirements",
                ),
            ),
            patch.object(memory.embedding, "embed_query", return_value=_vector()),
            patch.object(memory.vectorstore, "search_memory", return_value=stored),
        ):
            response = Agent(principal=ADMIN, llm_client=llm).run(
                "What are the Drive Agent assignment requirements?"
            )

        self.assertEqual(response, "Đã hiểu.")
        self.assertEqual([entry["tool"] for entry in AUDIT_LOG], ["search_memory"])
        self.assertIn("Drive Agent", llm.requests[0]["system_prompt"])
        tool_names = {tool["name"] for tool in llm.requests[0]["tools"]}
        self.assertNotIn("list_drive_files", tool_names)
        self.assertNotIn("search_drive_files", tool_names)

    def test_internal_upsert_rejects_model_initiated_calls(self):
        registry = ToolRegistry()
        registry.register(memory.upsert_user_memory_tool)

        response = registry.call(
            "upsert_user_memory",
            {
                "content": "The user likes Python.",
                "category": "user_preference",
                "topic": "programming_language",
                "value": "Python",
                "canonical_value": "python",
                "polarity": "like",
                "confidence": 0.99,
            },
            STANDARD_USER,
            model_initiated=True,
        )

        self.assertEqual(response["error_type"], "PermissionError")
        self.assertIn("internal", response["error"])


class EmbeddingServiceTests(unittest.TestCase):
    def tearDown(self):
        embedding._get_client.cache_clear()

    def test_fci_provider_uses_fci_key_and_base_url(self):
        with (
            patch.multiple(
                embedding,
                EMBEDDING_PROVIDER="fci",
                FCI_API_KEY="fci-test-key",
                FCI_BASE_URL="https://fci.example/v1",
            ),
            patch.object(embedding, "OpenAI") as openai_client,
        ):
            embedding._get_client.cache_clear()
            result = embedding._get_client()

        self.assertEqual(result, openai_client.return_value)
        openai_client.assert_called_once_with(
            api_key="fci-test-key",
            base_url="https://fci.example/v1",
        )

    def test_e5_model_uses_query_and_passage_prefixes(self):
        with patch.object(embedding, "EMBEDDING_MODEL", "multilingual-e5-large"):
            self.assertEqual(
                embedding._prepare_inputs(["Python"], query=False),
                ["passage: Python"],
            )
            self.assertEqual(
                embedding._prepare_inputs(["Python"], query=True),
                ["query: Python"],
            )


class MemoryToolTests(unittest.TestCase):
    def setUp(self):
        AUDIT_LOG.clear()
        rate_limiter._buckets.clear()

    def test_long_content_is_chunked_and_tagged_with_authenticated_user(self):
        content = "Python is preferred. " * 300
        captured = []

        def fake_save(records):
            captured.extend(records)
            return [{"id": str(index)} for index, _ in enumerate(records)]

        with (
            patch.object(
                memory.embedding,
                "embed_texts",
                side_effect=lambda chunks: [_vector() for _ in chunks],
            ),
            patch.object(memory.vectorstore, "save_memories", side_effect=fake_save),
        ):
            registry = ToolRegistry()
            registry.register(memory.save_document_memory_tool)
            response = registry.call(
                "save_document_memory",
                {"content": content, "category": "document"},
                STANDARD_USER,
            )

        self.assertNotIn("error", response)
        self.assertGreater(response["result"]["chunks_saved"], 1)
        self.assertTrue(
            all(
                len(memory._encode(record[0])) <= memory.MEMORY_CHUNK_TOKENS
                for record in captured
            )
        )
        self.assertTrue(
            all(
                record[2]["token_count"] == len(memory._encode(record[0]))
                for record in captured
            )
        )
        self.assertTrue(
            all(record[2]["user_id"] == "user_standard" for record in captured)
        )
        self.assertEqual(
            {record[2]["source_id"] for record in captured},
            {response["result"]["source_id"]},
        )

    def test_short_fact_is_stored_verbatim_as_one_chunk(self):
        content = "Tôi thích Python"
        chunks = memory._build_chunks(content, "user_preference")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, content)
        self.assertEqual(chunks[0].heading_paths, ())

    def test_document_preserves_markdown_blocks_and_heading_context(self):
        document = """# Python

Đoạn giới thiệu về Python.

## FastAPI

- Hỗ trợ async
- Tự sinh OpenAPI

| Tính năng | Trạng thái |
| --- | --- |
| Async | Có |

```python
app = FastAPI()
```
"""
        chunks = memory._build_chunks(document, "document")

        self.assertEqual(len(chunks), 1)
        self.assertIn("# Python\n## FastAPI", chunks[0].text)
        self.assertIn("- Hỗ trợ async\n- Tự sinh OpenAPI", chunks[0].text)
        self.assertIn("| Tính năng | Trạng thái |", chunks[0].text)
        self.assertIn("```python\napp = FastAPI()\n```", chunks[0].text)
        self.assertIn("Python > FastAPI", chunks[0].heading_paths)

    def test_only_oversized_block_is_hard_split_and_repeats_heading(self):
        document = "# Long section\n\n" + ("nội dung rất dài " * 900)
        chunks = memory._build_chunks(document, "document")

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.token_count <= memory.MEMORY_CHUNK_TOKENS for chunk in chunks)
        )
        self.assertTrue(all(chunk.text.startswith("# Long section") for chunk in chunks))
        self.assertTrue(
            all("Long section" in chunk.heading_paths for chunk in chunks)
        )

    def test_adjacent_chunks_have_configured_token_overlap(self):
        paragraphs = [
            f"Paragraph {index}: " + (f"word{index} " * 100)
            for index in range(8)
        ]
        chunks = memory._build_chunks("\n\n".join(paragraphs), "document")

        self.assertGreater(len(chunks), 1)
        previous_tail = memory._encode(chunks[0].text)[
            -memory.MEMORY_CHUNK_OVERLAP_TOKENS:
        ]
        self.assertEqual(
            memory._encode(chunks[1].text)[:len(previous_tail)],
            previous_tail,
        )

    def test_search_is_scoped_to_authenticated_user(self):
        with (
            patch.object(memory.embedding, "embed_query", return_value=_vector()),
            patch.object(memory.vectorstore, "search_memory", return_value=[]) as search,
        ):
            registry = ToolRegistry()
            registry.register(memory.search_memory_tool)
            response = registry.call(
                "search_memory",
                {"query": "What language do I like?"},
                STANDARD_USER,
            )

        self.assertEqual(response["result"]["total_results"], 0)
        self.assertEqual(search.call_args.kwargs["user_id"], "user_standard")
        self.assertEqual(
            search.call_args.kwargs["query_text"],
            "What language do I like?",
        )


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=vectorstore.MEMORY_COLLECTION,
            vectors_config=vectorstore.models.VectorParams(
                size=EMBEDDING_DIM,
                distance=vectorstore.models.Distance.COSINE,
            ),
        )
        self.client_patch = patch.object(
            vectorstore,
            "_get_client",
            return_value=self.client,
        )
        self.client_patch.start()

    def tearDown(self):
        self.client_patch.stop()
        self.client.close()

    def test_memory_survives_new_calls_and_hybrid_search_uses_raw_text(self):
        vectorstore.save_memory(
            "The user prefers Python for backend work.",
            _vector(0.8, 0.2),
            {"user_id": "alice", "category": "user_preference"},
        )
        vectorstore.save_memory(
            "Java deployment notes.",
            _vector(0.0, 1.0),
            {"user_id": "alice", "category": "document"},
        )
        vectorstore.save_memory(
            "Bob prefers Python.",
            _vector(1.0, 0.0),
            {"user_id": "bob"},
        )

        with patch.object(
            vectorstore,
            "_scroll_all",
            side_effect=AssertionError("recall must not scroll the corpus"),
        ):
            results = vectorstore.search_memory(
                _vector(0.0, 1.0),
                top_k=2,
                query_text="Python preference",
                user_id="alice",
            )

        self.assertEqual(len(results), 2)
        self.assertIn("Python", results[0]["text"])
        self.assertTrue(
            all(item["metadata"]["user_id"] == "alice" for item in results)
        )
        self.assertEqual(len(vectorstore.list_all_memories(user_id="alice")), 2)
        preferences = vectorstore.list_all_memories(
            user_id="alice",
            categories={"user_preference"},
        )
        self.assertEqual(len(preferences), 1)
        self.assertIn("Python", preferences[0]["text"])

    def test_inactive_points_are_filtered_before_hybrid_retrieval(self):
        vectorstore.save_memory(
            "Active Python preference.",
            _vector(),
            {"user_id": "alice", "category": "fact", "active": True},
        )
        vectorstore.save_memory(
            "Inactive Python preference.",
            _vector(),
            {"user_id": "alice", "category": "fact", "active": False},
        )

        results = vectorstore.search_memory(
            _vector(),
            top_k=5,
            query_text="Python preference",
            user_id="alice",
        )

        self.assertEqual(
            [item["text"] for item in results],
            ["Active Python preference."],
        )
        self.assertTrue(results[0]["lexical_match"])

    def test_existing_collection_is_upgraded_with_sparse_schema(self):
        vectorstore.ensure_collection()

        collection = self.client.get_collection(vectorstore.MEMORY_COLLECTION)

        self.assertIn(
            vectorstore.BM25_VECTOR_NAME,
            collection.config.params.sparse_vectors,
        )
        self.assertEqual(
            collection.config.params.sparse_vectors[
                vectorstore.BM25_VECTOR_NAME
            ].modifier,
            vectorstore.models.Modifier.IDF,
        )

    def test_hybrid_migration_is_idempotent_and_hides_legacy_preferences(self):
        vectorstore.ensure_collection()
        self.client.upsert(
            collection_name=vectorstore.MEMORY_COLLECTION,
            points=[
                vectorstore.models.PointStruct(
                    id=1,
                    vector=_vector(),
                    payload={
                        "text": "Alice likes legacy Python.",
                        "metadata": {
                            "user_id": "alice",
                            "category": "user_preference",
                        },
                    },
                ),
                vectorstore.models.PointStruct(
                    id=2,
                    vector=_vector(),
                    payload={
                        "text": "Alice does not like Python.",
                        "metadata": {
                            "user_id": "alice",
                            "category": "user_preference",
                            "memory_type": "preference",
                            "active": True,
                        },
                    },
                ),
            ],
        )

        first = migrate_qdrant_hybrid.migrate(batch_size=2)
        second = migrate_qdrant_hybrid.migrate(batch_size=2)
        points = {
            point.id: point
            for point in self.client.retrieve(
                collection_name=vectorstore.MEMORY_COLLECTION,
                ids=[1, 2],
                with_payload=True,
                with_vectors=True,
            )
        }

        self.assertEqual(first["backfilled_sparse"], 1)
        self.assertEqual(first["remaining_sparse"], 0)
        self.assertEqual(second["backfilled_sparse"], 0)
        self.assertFalse(points[1].payload["metadata"]["active"])
        self.assertTrue(points[1].payload["metadata"]["legacy_hidden"])
        self.assertIn(vectorstore.BM25_VECTOR_NAME, points[2].vector)

    def test_single_chunk_memory_key_is_idempotent(self):
        metadata = {
            "user_id": "alice",
            "category": "user_preference",
            "chunk_count": 1,
            "memory_key": "same-fact",
        }
        first = vectorstore.save_memory("Alice likes Python.", _vector(), metadata)
        second = vectorstore.save_memory("Alice likes Python.", _vector(), metadata)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(vectorstore.list_all_memories(user_id="alice")), 1)

    def test_structured_preference_polarity_change_updates_same_record(self):
        base_metadata = {
            "user_id": "alice",
            "category": "user_preference",
            "memory_type": "preference",
            "topic": "programming_language",
            "canonical_value": "python",
            "active": True,
            "chunk_count": 1,
            "memory_key": "preference-programming-language-python",
        }
        first = vectorstore.save_memory(
            "Alice likes Python.",
            _vector(),
            {**base_metadata, "polarity": "like"},
        )
        second = vectorstore.save_memory(
            "Alice does not like Python.",
            _vector(),
            {**base_metadata, "polarity": "dislike"},
        )

        current = vectorstore.list_all_memories(
            user_id="alice",
            categories={"user_preference"},
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["text"], "Alice does not like Python.")
        self.assertEqual(current[0]["metadata"]["polarity"], "dislike")

    def test_structured_preferences_hide_legacy_append_only_preferences(self):
        vectorstore.save_memory(
            "Alice likes legacy Python.",
            _vector(),
            {"user_id": "alice", "category": "user_preference"},
        )
        vectorstore.save_memory(
            "Alice does not like Python.",
            _vector(),
            {
                "user_id": "alice",
                "category": "user_preference",
                "memory_type": "preference",
                "topic": "programming_language",
                "canonical_value": "python",
                "polarity": "dislike",
                "active": True,
                "chunk_count": 1,
                "memory_key": "programming-language-python",
            },
        )

        current = vectorstore.list_all_memories(
            user_id="alice",
            categories={"user_preference"},
        )
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["metadata"]["polarity"], "dislike")

    def test_multichunk_document_upsert_is_idempotent(self):
        records = [
            (
                f"Document chunk {index}",
                _vector(),
                {
                    "user_id": "alice",
                    "category": "document",
                    "memory_key": "same-document",
                    "chunk_index": index,
                    "chunk_count": 2,
                },
            )
            for index in range(2)
        ]

        first = vectorstore.save_memories(records)
        second = vectorstore.save_memories(records)

        self.assertEqual(
            [item["id"] for item in first],
            [item["id"] for item in second],
        )
        self.assertEqual(len(vectorstore.list_all_memories(user_id="alice")), 2)


if __name__ == "__main__":
    unittest.main()
