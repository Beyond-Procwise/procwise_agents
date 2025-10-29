import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from botocore.exceptions import ClientError
from qdrant_client import models
from sentence_transformers import CrossEncoder

from services.rag_service import RAGService
from .base_agent import AgentOutput, AgentStatus, BaseAgent
from utils.gpu import configure_gpu, load_cross_encoder

configure_gpu()


logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Retrieval augmented generation agent with GPU optimisations."""

    _KNOWLEDGE_DOCUMENT_TYPES: Tuple[str, ...] = (
        "knowledge_graph_relation",
        "knowledge_graph_path",
        "supplier_flow",
        "supplier_relationship",
        "supplier_relationship_overview",
        "procurement_flow",
        "agent_relationship_summary",
    )

    _STRUCTURED_DOCUMENT_TYPES: Tuple[str, ...] = (
        "spend_summary",
        "contract_metadata",
        "contract_register",
        "supplier_kpi",
        "supplier_scorecard",
        "savings_pipeline",
        "category_savings",
        "invoice_register",
        "payment_terms",
    )

    _MEMORY_DOCUMENT_TYPES: Tuple[str, ...] = (
        "learning",
        "conversation_memory",
        "qa_memory",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        model_name = getattr(
            self.settings, "reranker_model", "BAAI/bge-reranker-large"
        )
        # Cross encoder ensures accurate ranking while utilising the GPU
        self._reranker = load_cross_encoder(
            model_name, CrossEncoder, getattr(self.agent_nick, "device", None)
        )
        # Thread pool enables parallel answer and follow-up generation
        self._executor = ThreadPoolExecutor(max_workers=2)
        # Service providing FAISS and BM25 retrieval
        self.rag_service = RAGService(agent_nick)

    def run(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        doc_type: str | None = None,
        product_type: str | None = None,
    ):
        """Answer questions while maintaining (but not utilising) chat history."""

        print(
            f"RAGAgent received query: '{query}' for user: '{user_id}'"
            + (f" in session '{session_id}'" if session_id else "")
        )

        history = self._load_chat_history(user_id, session_id)

        must: list[models.FieldCondition] = []
        if doc_type:
            must.append(
                models.FieldCondition(
                    key="document_type", match=models.MatchValue(value=doc_type.lower())
                )
            )
        if product_type:
            must.append(
                models.FieldCondition(
                    key="product_type", match=models.MatchValue(value=product_type.lower())
                )
            )
        qdrant_filter = models.Filter(must=must) if must else None

        search_hits = []
        seen_ids = set()
        query_variants = [query] + self._expand_query(query)
        for q in query_variants:
            hits = self.rag_service.search(q, top_k=top_k, filters=qdrant_filter)
            for h in hits:
                if h.id not in seen_ids:
                    search_hits.append(h)
                    seen_ids.add(h.id)

        learning_hits = self._search_learning_context(query_variants, top_k)
        for hit in learning_hits:
            if hit.id not in seen_ids:
                search_hits.append(hit)
                seen_ids.add(hit.id)

        kg_collection = getattr(self.settings, "knowledge_graph_collection_name", None)
        if kg_collection:
            for q in query_variants:
                kg_hits = self._search_knowledge_graph(q, top_k=top_k, filters=qdrant_filter)
                for hit in kg_hits:
                    if hit.id not in seen_ids:
                        search_hits.append(hit)
                        seen_ids.add(hit.id)

        if not doc_type and not any(
            (getattr(hit, "payload", {}) or {}).get("document_type") == "supplier_relationship"
            for hit in search_hits
        ):
            supplier_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_type", match=models.MatchValue(value="supplier_relationship")
                    )
                ]
            )
            targeted_hits = self.rag_service.search(query, top_k=top_k, filters=supplier_filter)
            for hit in targeted_hits:
                if hit.id not in seen_ids:
                    search_hits.append(hit)
                    seen_ids.add(hit.id)

        (
            structured_docs,
            narrative_docs,
            knowledge_hits,
            memory_docs,
        ) = self._categorize_hits(search_hits)

        structured_ranked = (
            self._rerank(query, structured_docs, top_k) if structured_docs else []
        )
        narrative_ranked = (
            self._rerank(query, narrative_docs, top_k) if narrative_docs else []
        )
        memory_ranked = (
            self._rerank(query, memory_docs, min(top_k, 3)) if memory_docs else []
        )

        reranked_docs = self._merge_ranked_layers(
            structured_ranked, narrative_ranked, top_k
        )
        supplementary_narrative = [
            hit
            for hit in narrative_ranked
            if hit not in reranked_docs
        ][: max(0, top_k - len(reranked_docs))]
        full_document_set = reranked_docs + supplementary_narrative
        structured_ids = {getattr(hit, "id", id(hit)) for hit in structured_ranked}
        narrative_focus = [
            hit
            for hit in full_document_set
            if getattr(hit, "id", id(hit)) not in structured_ids
        ]

        knowledge_hits_sorted = (
            sorted(knowledge_hits, key=lambda h: getattr(h, "score", 0.0), reverse=True)[
                :top_k
            ]
            if knowledge_hits
            else []
        )

        if not full_document_set and not knowledge_hits_sorted and not memory_ranked:
            fallback_answer = (
                "I could not find any relevant documents to answer your question."
            )
            default_followups = [
                "Can you share additional details or data sources to search?",
                "Should I widen the analysis to other spend categories?",
                "Do you want me to escalate this to procurement operations?",
            ]
            formatted_answer = self._compose_final_answer(
                query, fallback_answer, default_followups
            )
            primary_topic = "general"
            history.append(
                {
                    "query": query,
                    "answer": formatted_answer,
                    "topic": primary_topic,
                    "follow_up_questions": default_followups,
                }
            )
            self._save_chat_history(user_id, history, session_id)
            return {
                "answer": formatted_answer,
                "follow_up_questions": default_followups,
                "retrieved_documents": [],
            }

        structured_context = self._build_document_context(structured_ranked)
        narrative_context = self._build_document_context(narrative_focus)
        memory_context = self._build_document_context(memory_ranked)
        knowledge_summary, knowledge_payloads = self._build_knowledge_summary(
            knowledge_hits_sorted
        )
        outline = self._build_context_outline(full_document_set, knowledge_summary)
        plan = self._generate_agentic_plan(query, outline, knowledge_summary)

        plan_text = plan.strip() if plan else (
            "1. Review knowledge graph coverage to identify relevant suppliers.\n"
            "2. Inspect retrieved documents for concrete figures and policy references.\n"
            "3. Draft the answer with citations and note outstanding follow-up items."
        )
        outline_text = outline.strip() if outline else (
            "- Prioritise suppliers and procurement records that align with the question."
        )

        retrieved_sections: List[str] = []
        if knowledge_summary:
            retrieved_sections.append(
                f"KNOWLEDGE GRAPH INSIGHTS:\n{knowledge_summary.strip()}"
            )
        if structured_context:
            retrieved_sections.append(
                f"STRUCTURED DATA:\n{structured_context.strip()}"
            )
        if narrative_context and narrative_context != structured_context:
            retrieved_sections.append(
                f"NARRATIVE INSIGHTS:\n{narrative_context.strip()}"
            )
        if memory_context:
            retrieved_sections.append(
                f"MEMORY NOTES:\n{memory_context.strip()}"
            )

        context_block = "\n\n".join(retrieved_sections).strip()
        if not context_block:
            payload_snapshot = [
                getattr(hit, "payload", {})
                for hit in full_document_set + knowledge_hits_sorted + memory_ranked
            ]
            context_block = self._truncate_text(
                json.dumps(payload_snapshot, ensure_ascii=False)
            )

        context_block = self._limit_context(context_block)

        conversation_context = self._format_recent_history(history, limit=3)
        if conversation_context:
            conversation_section = f"RECENT CONVERSATION:\n{conversation_context}\n\n"
        else:
            conversation_section = ""

        rag_prompt = (
            "You are the ProcWise Procurement Intelligence Assistant. Use ONLY the provided RETRIEVED CONTENT to answer "
            "the USER QUESTION. If information is missing, acknowledge the gap explicitly.\n\n"
            "Instructions:\n"
            "1) Respond with a concise procurement analysis (2-3 sentences) that delivers a complete, decision-ready insight.\n"
            "2) Maintain a professional, data-aware tone.\n"
            "3) Preserve numeric values, dates, currencies, and units exactly as presented.\n"
            "4) If the context does not answer the question, reply exactly with: Unable to answer from the provided materials.\n"
            "5) Do not fabricate suppliers, contracts, figures, or policies beyond the context.\n\n"
            f"AGENTIC PLAN:\n{plan_text}\n\n"
            f"CONTEXT OUTLINE:\n{outline_text}\n\n"
            f"{conversation_section}RETRIEVED CONTENT:\n{context_block}\n\nUSER QUESTION: {query}\n\nReturn only the final answer string below:\n"
        )

        # Generate answer and follow-up suggestions in parallel
        answer_model = getattr(
            self.settings, "rag_model", getattr(self.settings, "extraction_model", None)
        )
        answer_future = self._executor.submit(
            self.call_ollama, rag_prompt, model=answer_model
        )
        combined_context = "\n\n".join(
            part
            for part in (
                outline,
                knowledge_summary,
                structured_context,
                narrative_context,
                memory_context,
            )
            if part
        )
        follow_future = self._executor.submit(
            self._generate_followups, query, combined_context
        )

        answer_resp = answer_future.result()
        followups = follow_future.result()
        answer = answer_resp.get("response", "I am sorry, I could not generate an answer.")
        normalised_followups = self._normalise_followups(
            query,
            followups,
            structured_ranked,
            knowledge_hits_sorted,
            memory_ranked,
        )
        final_answer = self._compose_final_answer(query, answer, normalised_followups)
        primary_topic = self._determine_primary_topic(
            full_document_set, knowledge_hits_sorted, memory_ranked
        )

        history.append(
            {
                "query": query,
                "answer": final_answer,
                "topic": primary_topic,
                "follow_up_questions": normalised_followups,
            }
        )
        self._save_chat_history(user_id, history, session_id)

        retrieved_payloads: List[Dict[str, Any]] = []
        for hit in full_document_set:
            payload = getattr(hit, "payload", {}) or {}
            payload_copy = dict(payload)
            payload_copy.setdefault("topic", self._infer_topic(payload_copy))
            retrieved_payloads.append(payload_copy)
        for hit in memory_ranked:
            payload = getattr(hit, "payload", {}) or {}
            payload_copy = dict(payload)
            payload_copy.setdefault("topic", self._infer_topic(payload_copy))
            retrieved_payloads.append(payload_copy)
        for payload in knowledge_payloads:
            payload.setdefault("topic", self._infer_topic(payload))
        retrieved_payloads.extend(knowledge_payloads)
        if outline:
            retrieved_payloads.append(
                {"document_type": "context_outline", "outline": outline}
            )
        if knowledge_summary:
            retrieved_payloads.append(
                {
                    "document_type": "knowledge_summary",
                    "summary": knowledge_summary,
                    "topic": self._infer_topic({"document_type": "knowledge_summary"}),
                }
            )
        result = AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "answer": final_answer,
                "follow_up_questions": normalised_followups,
                "retrieved_documents": retrieved_payloads,
            },
            agentic_plan=plan_text,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rerank(self, query: str, hits, top_k: int):
        """Re-rank search hits using a cross-encoder for improved accuracy."""
        if not hits:
            return []
        pairs = [
            (
                query,
                hit.payload.get(
                    "text_summary",
                    hit.payload.get(
                        "content",
                        hit.payload.get(
                            "summary", json.dumps(hit.payload, ensure_ascii=False)
                        ),
                    ),
                ),
            )
            for hit in hits
        ]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        filtered = [hit for hit, score in ranked if score > 0][:top_k]
        if filtered:
            return filtered
        return [hit for hit, _ in ranked[:top_k]]

    def _merge_ranked_layers(
        self, structured_hits: Sequence[Any], narrative_hits: Sequence[Any], limit: int
    ) -> List[Any]:
        """Merge structured and narrative hits giving priority to structured results."""

        seen: set[Any] = set()
        merged: List[Any] = []
        for layer in (structured_hits, narrative_hits):
            for hit in layer:
                hit_id = getattr(hit, "id", None)
                if hit_id is None:
                    hit_id = id(hit)
                if hit_id in seen:
                    continue
                merged.append(hit)
                seen.add(hit_id)
                if len(merged) >= limit:
                    return merged[:limit]
        return merged[:limit]

    def _is_structured_payload(self, payload: Dict[str, Any]) -> bool:
        """Heuristically determine if the payload represents structured data."""

        doc_type = str(payload.get("document_type") or "").lower()
        if doc_type in self._STRUCTURED_DOCUMENT_TYPES:
            return True
        structured_keys = {
            "total_spend",
            "total_value",
            "total_value_gbp",
            "savings",
            "savings_value",
            "contract_value",
            "contract_end_date",
            "kpi_score",
            "rows",
            "table",
            "metrics",
        }
        if any(key in payload for key in structured_keys):
            return True
        if isinstance(payload.get("rows"), list):
            return True
        table_value = payload.get("table")
        if isinstance(table_value, (list, dict)):
            return True
        return False

    def _limit_context(self, text: str, max_chars: int | None = None) -> str:
        """Enforce a maximum character length for the retrieval context."""

        if not text:
            return ""
        limit = max_chars or getattr(self.settings, "rag_context_chars", 6000)
        limit = max(1000, int(limit))
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        if "\n" in truncated:
            truncated = truncated.rsplit("\n", 1)[0]
        return truncated.strip()

    def _compose_final_answer(
        self, query: str, answer: str, followups: Sequence[str]
    ) -> str:
        """Format the final response using the ProcWise Q&A template."""

        clean_answer = (answer or "Unable to answer from the provided materials.").strip()
        if not clean_answer:
            clean_answer = "Unable to answer from the provided materials."
        lines = [f"Q: {query.strip()}", "", f"A: {clean_answer}", "", "Suggested Follow-Up Prompts:"]
        for item in followups:
            lines.append(f"- {item.strip()}")
        return "\n".join(lines).strip()

    def _normalise_followups(
        self,
        query: str,
        followups: Sequence[str],
        structured_hits: Sequence[Any],
        knowledge_hits: Sequence[Any],
        memory_hits: Sequence[Any],
    ) -> List[str]:
        """Ensure exactly three high-quality follow-up questions."""

        cleaned: List[str] = []
        for item in followups or []:
            if not isinstance(item, str):
                continue
            candidate = item.strip().lstrip("-• ").strip()
            if not candidate:
                continue
            if candidate not in cleaned:
                cleaned.append(candidate)
            if len(cleaned) == 3:
                break

        topic = self._determine_primary_topic(structured_hits, knowledge_hits, memory_hits)
        template_candidates = self._topic_followups(topic)
        for candidate in template_candidates:
            if len(cleaned) >= 3:
                break
            if candidate not in cleaned:
                cleaned.append(candidate)

        while len(cleaned) < 3:
            fallback = "What additional detail would help you act on this insight?"
            if fallback in cleaned:
                fallback = f"Which next step should we take for {topic or 'this area'}?"
            cleaned.append(fallback)

        return cleaned[:3]

    def _topic_followups(self, topic: str | None) -> List[str]:
        """Return template follow-ups aligned to the inferred topic."""

        templates: Dict[str, List[str]] = {
            "savings": [
                "How does this compare with last year's savings delivery?",
                "Which categories contributed most to the savings so far?",
                "What savings remain in the pipeline and when will they land?",
            ],
            "supplier": [
                "Are there performance concerns or KPIs trending negatively?",
                "Do we have alternative suppliers to diversify this category?",
                "Should we schedule a supplier review or QBR?",
            ],
            "contracts": [
                "Which contracts are up for renewal in the next quarter?",
                "Do any terms require CFO exception or escalation?",
                "Are we aligned on compliance with agreed payment terms?",
            ],
            "spend": [
                "What is the trend versus last year for this spend area?",
                "Which suppliers drive the majority of this spend?",
                "Do we have opportunities to consolidate vendors?",
            ],
            "compliance": [
                "Are there open actions to remediate the compliance gaps?",
                "Who owns the follow-up for policy adherence?",
                "Do we need to escalate this risk to governance?",
            ],
            "risk": [
                "What mitigation steps are in place for the highlighted risk?",
                "Is there exposure to supplier failure or service disruption?",
                "Should we update the risk register for this item?",
            ],
        }
        if topic and topic in templates:
            return templates[topic]
        # Default procurement follow-ups
        return [
            "Do you want to drill into the category drivers behind this insight?",
            "Should we review supplier performance before the next cycle?",
            "Would trend analysis over the past year be helpful?",
        ]

    def _determine_primary_topic(
        self,
        docs: Sequence[Any],
        knowledge_hits: Sequence[Any],
        memory_hits: Sequence[Any],
    ) -> str:
        """Infer the dominant topic from retrieved payloads."""

        topics: List[str] = []
        for collection in (docs, knowledge_hits, memory_hits):
            for hit in collection:
                payload = getattr(hit, "payload", {}) or {}
                topic = self._infer_topic(payload)
                if topic:
                    topics.append(topic)
        if not topics:
            return "general"
        counter = Counter(topics)
        return counter.most_common(1)[0][0]

    def _infer_topic(self, payload: Dict[str, Any]) -> str:
        """Map document metadata to a procurement topic label."""

        doc_type = str(payload.get("document_type") or "").lower()
        topic_map = {
            "savings_pipeline": "savings",
            "category_savings": "savings",
            "savings_summary": "savings",
            "spend_summary": "spend",
            "spend_analysis": "spend",
            "supplier_kpi": "supplier",
            "supplier_scorecard": "supplier",
            "supplier_relationship": "supplier",
            "supplier_flow": "supplier",
            "contract_metadata": "contracts",
            "contract_register": "contracts",
            "payment_terms": "contracts",
            "policy": "compliance",
            "compliance": "compliance",
            "risk_register": "risk",
            "risk": "risk",
            "knowledge_summary": "knowledge",
        }
        if doc_type in topic_map:
            return topic_map[doc_type]
        if "savings" in doc_type:
            return "savings"
        if any(keyword in doc_type for keyword in ("supplier", "kpi")):
            return "supplier"
        if "contract" in doc_type:
            return "contracts"
        if "policy" in doc_type or "compliance" in doc_type:
            return "compliance"
        if "spend" in doc_type:
            return "spend"
        if "risk" in doc_type:
            return "risk"
        if "invoice" in doc_type:
            return "spend"
        if "learning" in doc_type or payload.get("source") == "memory":
            return "memory"
        return "general"

    def _expand_query(self, query: str) -> list:
        """Generate alternative phrasings to boost retrieval recall."""
        prompt = (
            "Provide up to three alternate phrasings or related search terms for the "
            "following procurement question. Return JSON {\"expansions\": [\"...\"]}.\n"
            f"QUESTION: {query}"
        )
        try:  # pragma: no cover - network call
            resp = self.call_ollama(
                prompt, model=self.settings.extraction_model, format="json"
            )
            data = json.loads(resp.get("response", "{}"))
            expansions = data.get("expansions", [])
            if isinstance(expansions, list):
                return [e for e in expansions if isinstance(e, str)]
        except Exception:
            pass
        return []

    def _generate_followups(self, query: str, context: str):
        """Generate follow-up questions based on document and knowledge context."""
        prompt = (
            "You are a helpful procurement assistant. Based on the user's question and the "
            "retrieved context (which can include knowledge graph insights, supplier flows "
            "and document excerpts), suggest three concise follow-up questions that would "
            "clarify the request or gather more details. Return each question on a new line.\n\n"
            f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nFOLLOW_UP_QUESTIONS:"
        )
        resp = self.call_ollama(prompt, model=self.settings.extraction_model)
        questions = resp.get("response", "").strip().splitlines()
        return [q for q in questions if q]

    def _format_recent_history(self, history: Sequence[Dict[str, Any]], limit: int = 3) -> str:
        """Return the last ``limit`` user/assistant turns as a formatted string."""

        if not history:
            return ""

        recent_turns = history[-max(1, limit) :]
        lines: List[str] = []
        for turn in recent_turns:
            if not isinstance(turn, dict):
                continue
            query = turn.get("query")
            if isinstance(query, str) and query.strip():
                lines.append(f"User: {query.strip()}")
            answer = turn.get("answer")
            if isinstance(answer, str) and answer.strip():
                lines.append(f"Assistant: {answer.strip()}")
        return "\n".join(lines).strip()

    def _load_chat_history(self, user_id: str, session_id: str | None = None) -> list:
        history_key = (
            f"chat_history/{user_id}/{session_id}.json" if session_id else f"chat_history/{user_id}.json"
        )
        try:
            with self._borrow_s3_client() as s3_client:
                s3_object = s3_client.get_object(
                    Bucket=self.settings.s3_bucket_name, Key=history_key
                )
            body = s3_object.get("Body")
            if body is None:
                print(
                    f"No body returned for chat history '{history_key}'. Starting new session."
                )
                return []
            try:
                history = json.loads(body.read().decode("utf-8"))
            finally:
                try:
                    body.close()
                except Exception:
                    logger.debug("Failed to close chat history stream %s", history_key, exc_info=True)
            print(
                f"Loaded {len(history)} items from chat history for user '{user_id}'"
                + (f" session '{session_id}'" if session_id else "")
                + "."
            )
            return history
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print(
                    f"No chat history found for user '{user_id}'"
                    + (f" session '{session_id}'" if session_id else "")
                    + ". Starting new session."
                )
                return []
            raise

    def _save_chat_history(
        self, user_id: str, history: list, session_id: str | None = None
    ) -> None:
        history_key = (
            f"chat_history/{user_id}/{session_id}.json" if session_id else f"chat_history/{user_id}.json"
        )
        bucket_name = getattr(self.settings, "s3_bucket_name", None)
        if not bucket_name:
            print(
                "ERROR: S3 bucket name is not configured. Please set 's3_bucket_name' in settings."
            )
            return
        try:
            with self._borrow_s3_client() as s3_client:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=history_key,
                    Body=json.dumps(history, indent=2),
                )
            print(
                f"Saved chat history for user '{user_id}'"
                + (f" session '{session_id}'" if session_id else "")
                + f" in bucket '{bucket_name}'."
            )
        except ClientError as e:
            error_code = e.response["Error"].get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                print(
                    f"ERROR: S3 bucket '{bucket_name}' does not exist. Please check configuration."
                )
            else:
                print(
                    f"ERROR: Could not save chat history to S3 bucket '{bucket_name}'. {e}"
                )
        except Exception as e:
            print(
                f"ERROR: Could not save chat history to S3 bucket '{bucket_name}'. {e}"
            )

    def _categorize_hits(
        self, hits: Sequence[Any]
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """Separate hits into structured, narrative, knowledge, and memory layers."""

        structured: List[Any] = []
        narrative: List[Any] = []
        knowledge: List[Any] = []
        memory: List[Any] = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            doc_type = str(payload.get("document_type") or "").lower()
            if (
                doc_type in self._KNOWLEDGE_DOCUMENT_TYPES
                or doc_type.startswith("knowledge_graph")
                or doc_type.endswith("_flow")
                or (
                    not payload.get("content")
                    and not payload.get("summary")
                    and any(
                        key in payload
                        for key in (
                            "relationship_type",
                            "path",
                            "coverage_ratio",
                        )
                    )
                )
            ):
                knowledge.append(hit)
                continue
            if doc_type in self._MEMORY_DOCUMENT_TYPES or payload.get("source") == "memory":
                memory.append(hit)
                continue
            if self._is_structured_payload(payload):
                structured.append(hit)
            else:
                narrative.append(hit)
        return structured, narrative, knowledge, memory

    def _search_learning_context(
        self, queries: Sequence[str], top_k: int
    ) -> List[Any]:
        collection_name = getattr(self.settings, "learning_collection_name", None)
        if not collection_name:
            return []
        client = getattr(self.agent_nick, "qdrant_client", None)
        embedder = getattr(self.agent_nick, "embedding_model", None)
        if client is None or embedder is None or not hasattr(client, "search"):
            return []

        filter_learning = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_type", match=models.MatchValue(value="learning")
                )
            ]
        )
        collected: Dict[Any, Any] = {}
        for query in queries:
            if not query:
                continue
            try:
                vector = embedder.encode(query, normalize_embeddings=True)
            except Exception:
                logger.exception("Failed to encode learning context query")
                continue
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            search_params = models.SearchParams(hnsw_ef=128, exact=False)
            try:
                hits = client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    query_filter=filter_learning,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                )
            except Exception:
                logger.exception("Learning search failed for query '%s'", query)
                continue
            for hit in hits or []:
                hit_id = getattr(hit, "id", None)
                if hit_id is None or hit_id in collected:
                    continue
                collected[hit_id] = hit
        return list(collected.values())[:top_k]

    def _search_knowledge_graph(
        self, query: str, top_k: int = 5, filters: Optional[models.Filter] = None
    ):
        """Search the dedicated knowledge graph collection for supporting context."""

        collection = getattr(self.settings, "knowledge_graph_collection_name", None)
        client = getattr(self.agent_nick, "qdrant_client", None)
        embedder = getattr(self.agent_nick, "embedding_model", None)
        if not all([collection, client, embedder]) or not hasattr(client, "search"):
            return []
        try:
            vector = embedder.encode(query, normalize_embeddings=True).tolist()
        except Exception:
            logger.exception("Failed to encode query for knowledge graph search")
            return []

        search_params = models.SearchParams(hnsw_ef=256, exact=False)
        try:
            hits = client.search(
                collection_name=collection,
                query_vector=vector,
                query_filter=filters,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=search_params,
            )
        except Exception:
            logger.exception("Knowledge graph search failed")
            return []

        if hits:
            return hits

        try:
            exact_params = models.SearchParams(hnsw_ef=256, exact=True)
            return client.search(
                collection_name=collection,
                query_vector=vector,
                query_filter=filters,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=exact_params,
            )
        except Exception:
            logger.exception("Exact knowledge graph search failed")
            return []

    def _build_document_context(self, hits: Sequence[Any]) -> str:
        """Render a structured textual context from reranked document hits."""

        sections: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            payload: Dict[str, Any] = getattr(hit, "payload", {}) or {}
            doc_id = (
                payload.get("record_id")
                or payload.get("document_id")
                or payload.get("id")
                or hit.id
            )
            doc_type = str(payload.get("document_type") or "document")
            supplier = payload.get("supplier_name") or payload.get("supplier_id")
            header_parts = [f"Document {idx}", f"ID: {doc_id}", f"Type: {doc_type}"]
            if supplier:
                header_parts.append(f"Supplier: {supplier}")
            header = " | ".join(header_parts)
            content = (
                payload.get("content")
                or payload.get("summary")
                or payload.get("text")
            )
            if isinstance(content, (list, dict)):
                content = json.dumps(content, ensure_ascii=False)
            if not content:
                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "vector",
                        "embedding",
                        "content",
                        "summary",
                        "text",
                    }
                    and value not in (None, "")
                }
                content = json.dumps(metadata, ensure_ascii=False) if metadata else ""
            snippet = self._truncate_text(str(content).strip()) if content else ""
            sections.append(f"{header}\n{snippet}")
        return "\n\n".join(sections).strip()

    def _build_context_outline(
        self, document_hits: Sequence[Any], knowledge_summary: str
    ) -> str:
        """Generate a concise outline of the retrieved evidence."""

        lines: List[str] = []
        for hit in document_hits:
            payload: Dict[str, Any] = getattr(hit, "payload", {}) or {}
            doc_type = str(payload.get("document_type") or "document").replace("_", " ")
            doc_id = payload.get("record_id") or payload.get("document_id") or hit.id
            supplier = payload.get("supplier_name") or payload.get("supplier_id")
            summary = payload.get("summary") or payload.get("content")
            summary_text = ""
            if isinstance(summary, str):
                summary_text = summary.strip().splitlines()[0]
            elif isinstance(summary, (list, dict)):
                summary_text = json.dumps(summary, ensure_ascii=False)
            outline_line = f"- {doc_type.title()} {doc_id}"
            if supplier:
                outline_line += f" (supplier {supplier})"
            if summary_text:
                outline_line += f": {self._truncate_text(summary_text, 140)}"
            lines.append(outline_line)

        if knowledge_summary:
            first_sentence = knowledge_summary.strip().split(".")[0].strip()
            if first_sentence:
                lines.append(f"- Knowledge graph insight: {first_sentence}.")

        return "\n".join(lines).strip()

    def _build_knowledge_summary(
        self, hits: Sequence[Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Summarise knowledge graph hits into natural language text."""

        if not hits:
            return "", []

        summaries: List[str] = []
        payloads: List[Dict[str, Any]] = []
        for hit in hits:
            payload: Dict[str, Any] = dict(getattr(hit, "payload", {}) or {})
            if not payload:
                continue
            payloads.append(payload)
            doc_type = str(payload.get("document_type") or "").lower()
            if doc_type == "supplier_flow":
                supplier = payload.get("supplier_name") or payload.get("supplier_id")
                coverage = payload.get("coverage_ratio")
                parts: List[str] = []
                if supplier:
                    parts.append(str(supplier))
                if isinstance(coverage, (int, float)):
                    parts.append(f"coverage {coverage * 100:.1f}%")
                for key, label in (
                    ("contracts", "contracts"),
                    ("purchase_orders", "purchase orders"),
                    ("invoices", "invoices"),
                    ("quotes", "quotes"),
                ):
                    details = payload.get(key)
                    if isinstance(details, dict):
                        count = details.get("count")
                        if isinstance(count, (int, float)) and count:
                            parts.append(f"{int(count)} {label}")
                products = payload.get("products")
                if isinstance(products, dict):
                    top_items = products.get("top_items")
                    if isinstance(top_items, list) and top_items:
                        first = top_items[0]
                        if isinstance(first, dict) and first.get("description"):
                            parts.append(f"focus on {first['description']}")
                if parts:
                    summaries.append(
                        "Supplier flow: " + ", ".join(parts).rstrip(", ") + "."
                    )
                mapping_summary = payload.get("mapping_summary")
                if isinstance(mapping_summary, list):
                    for statement in mapping_summary:
                        if isinstance(statement, str) and statement.strip():
                            summaries.append(statement.strip().rstrip(".") + ".")
            elif doc_type == "supplier_relationship":
                statements = payload.get("relationship_statements")
                if isinstance(statements, list) and statements:
                    for statement in statements:
                        if isinstance(statement, str) and statement.strip():
                            summaries.append(statement.strip().rstrip(".") + ".")
                summary_text = payload.get("summary") or payload.get("content")
                if isinstance(summary_text, str) and summary_text.strip():
                    summaries.append(self._truncate_text(summary_text.strip()))
            elif doc_type == "supplier_relationship_overview":
                total_suppliers = payload.get("total_suppliers")
                average_cov = payload.get("average_coverage_ratio")
                line = "Supplier relationship overview"
                if isinstance(total_suppliers, (int, float)):
                    line += f" covering {int(total_suppliers)} suppliers"
                if isinstance(average_cov, (int, float)):
                    line += f" with average coverage {average_cov * 100:.1f}%"
                summaries.append(line.strip() + ".")
                statements = payload.get("relationship_statements")
                if isinstance(statements, list):
                    for statement in statements:
                        if isinstance(statement, str) and statement.strip():
                            summaries.append(statement.strip().rstrip(".") + ".")
                top_coverage = payload.get("top_suppliers_by_coverage")
                if isinstance(top_coverage, list) and top_coverage:
                    highlights: List[str] = []
                    for item in top_coverage[:3]:
                        if not isinstance(item, dict):
                            continue
                        supplier = item.get("supplier_name") or item.get("supplier_id")
                        coverage = item.get("coverage_ratio")
                        if supplier:
                            if isinstance(coverage, (int, float)):
                                highlights.append(f"{supplier} ({coverage * 100:.1f}% coverage)")
                            else:
                                highlights.append(str(supplier))
                    if highlights:
                        summaries.append("Highest coverage suppliers: " + ", ".join(highlights) + ".")
                top_spend = payload.get("top_suppliers_by_spend")
                if isinstance(top_spend, list) and top_spend:
                    spend_highlights: List[str] = []
                    for item in top_spend[:3]:
                        if not isinstance(item, dict):
                            continue
                        supplier = item.get("supplier_name") or item.get("supplier_id")
                        spend_value = item.get("total_value_gbp")
                        if supplier:
                            if isinstance(spend_value, (int, float)):
                                spend_highlights.append(f"{supplier} ({spend_value:,.2f} GBP)")
                            else:
                                spend_highlights.append(str(supplier))
                    if spend_highlights:
                        summaries.append("Top spend suppliers: " + ", ".join(spend_highlights) + ".")
            elif doc_type.startswith("knowledge_graph"):
                source = payload.get("source_table") or payload.get("path")
                relationship = payload.get("relationship_type") or "relationship"
                target = payload.get("target_table") or "target"
                confidence = payload.get("confidence")
                sentence = f"{source} {relationship} {target}" if source else relationship
                if isinstance(confidence, (int, float)):
                    sentence += f" (confidence {confidence:.2f})"
                description = payload.get("description")
                if description:
                    sentence += f" – {description}"
                summaries.append(sentence.strip() + ".")
            elif doc_type == "agent_relationship_summary":
                agent = payload.get("agent_name") or "Agent"
                agent_summary = payload.get("summary")
                if isinstance(agent_summary, str) and agent_summary.strip():
                    summaries.append(f"{agent}: {self._truncate_text(agent_summary.strip(), 280)}")
                statements = payload.get("relationship_statements")
                if isinstance(statements, list):
                    for statement in statements[:5]:
                        if isinstance(statement, str) and statement.strip():
                            summaries.append(statement.strip().rstrip(".") + ".")
                flow_snapshot = payload.get("flow_snapshot_size")
                if isinstance(flow_snapshot, (int, float)) and flow_snapshot:
                    summaries.append(
                        f"Knowledge graph snapshot covers {int(flow_snapshot)} supplier flow"
                        f"{'s' if int(flow_snapshot) != 1 else ''}."
                    )
            else:
                description = payload.get("description") or payload.get("summary")
                if description:
                    summaries.append(str(description).strip())

        summary_text = " ".join(summaries).strip()
        if summary_text:
            summary_text = self._truncate_text(summary_text)
        return summary_text, payloads

    def _generate_agentic_plan(
        self, query: str, outline: str, knowledge_summary: str
    ) -> str:
        """Create a lightweight reasoning plan for the answering model."""

        planning_prompt = (
            "You are coordinating an agentic RAG workflow for procurement analytics. "
            "Given the user's question, the context outline and knowledge graph insights, "
            "produce a short step-by-step plan (3-4 numbered steps) describing how to answer "
            "the question using the available evidence. Keep the plan succinct.\n\n"
            f"USER QUESTION: {query}\n\nCONTEXT OUTLINE:\n{outline or 'No outline provided.'}\n\n"
            f"KNOWLEDGE SUMMARY:\n{knowledge_summary or 'No knowledge summary available.'}"
        )
        model_name = getattr(self.settings, "planning_model", self.settings.extraction_model)
        try:  # pragma: no cover - network/LLM call
            response = self.call_ollama(planning_prompt, model=model_name)
            plan_text = response.get("response", "").strip()
            if plan_text:
                return plan_text
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Agentic planning generation failed: %s", exc)
        return (
            "1. Review knowledge graph coverage to identify relevant suppliers.\n"
            "2. Inspect retrieved documents for concrete figures and policy references.\n"
            "3. Draft the answer with citations and note outstanding follow-up items."
        )

    @staticmethod
    def _truncate_text(text: str, limit: int = 1200) -> str:
        """Trim long text segments while keeping them readable."""

        if len(text) <= limit:
            return text
        truncated = text[:limit].rstrip()
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        if not truncated:
            truncated = text[:limit]
        return truncated.rstrip("., ") + "..."
