import logging
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from datetime import datetime, time, timedelta, timezone
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import models

import services.rag_service as rag_module


logger = logging.getLogger(__name__)


class SupplierRelationshipService:
    def __init__(self, agent_nick) -> None:
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        try:
            self.rag_service = rag_module.RAGService(agent_nick)
        except Exception:
            logger.debug("SupplierRelationshipService failed to initialise RAGService", exc_info=True)
            self.rag_service = None
        self.collection_name = getattr(self.settings, "qdrant_collection_name", None)
        self.client = getattr(agent_nick, "qdrant_client", None)
        cache_limit = getattr(self.settings, "relationship_cache_limit", None)
        if not isinstance(cache_limit, int) or cache_limit <= 0:
            cache_limit = 512
        self._fallback_cache_limit = cache_limit
        self._fallback_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._fallback_negative_cache: OrderedDict[str, None] = OrderedDict()
        self._ensure_payload_indexes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index_supplier_flows(self, flows: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        valid_flows: List[Mapping[str, Any]] = []
        for flow in flows:
            summary, statements = self._summarise_flow(flow)
            if not summary:
                continue
            valid_flows.append(flow)
            supplier_id = flow.get("supplier_id")
            supplier_name = flow.get("supplier_name")
            record_id = self._record_id(supplier_id, supplier_name)
            metadata = {
                "record_id": record_id,
                "document_type": "supplier_relationship",
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "supplier_name_normalized": self._normalise_key(supplier_name),
                "coverage_ratio": flow.get("coverage_ratio"),
                "contracts": flow.get("contracts"),
                "purchase_orders": flow.get("purchase_orders"),
                "invoices": flow.get("invoices"),
                "quotes": flow.get("quotes"),
                "products": flow.get("products"),
                "relationship_statements": statements,
            }
            entries.append({"text": summary, "metadata": metadata})

        if not entries:
            return None

        overview_bundle = self._build_overview(valid_flows)

        if self.rag_service is None:
            return overview_bundle["metadata"] if overview_bundle else None

        for entry in entries:
            try:
                self.rag_service.upsert_texts([entry["text"]], metadata=entry["metadata"])
            except Exception:
                logger.exception(
                    "Failed to upsert supplier relationship for %s", entry["metadata"].get("supplier_id")
                )

        if overview_bundle:
            try:
                self.rag_service.upsert_texts(
                    [overview_bundle["text"]], metadata=overview_bundle["metadata"]
                )
            except Exception:
                logger.exception("Failed to upsert supplier relationship overview")

        return overview_bundle["metadata"] if overview_bundle else None

    def fetch_relationship(
        self,
        supplier_id: Optional[str] = None,
        supplier_name: Optional[str] = None,
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        if not self.client or not self.collection_name or not hasattr(self.client, "scroll"):
            return []

        must: List[models.FieldCondition] = []
        if supplier_id:
            must.append(
                models.FieldCondition(
                    key="supplier_id",
                    match=models.MatchValue(value=str(supplier_id)),
                )
            )
        if supplier_name:
            normalised = self._normalise_key(supplier_name)
            if normalised:
                must.append(
                    models.FieldCondition(
                        key="supplier_name_normalized",
                        match=models.MatchValue(value=normalised),
                    )
                )

        if not must:
            return []

        cached_payloads = self._lookup_cached_relationship(supplier_id, supplier_name, limit)
        if cached_payloads is not None:
            return cached_payloads

        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must),
                with_payload=True,
                with_vectors=False,
                limit=limit,
            )
        except Exception as exc:
            if self._is_missing_index_error(exc):
                logger.warning(
                    "SupplierRelationshipService: missing Qdrant index detected while loading supplier relationship. Attempting to create payload indexes before falling back.",
                    exc_info=True,
                )
                # Ensure indexes exist and retry once before resorting to a full collection scan.
                self._ensure_payload_indexes()
                cached_payloads = self._lookup_cached_relationship(supplier_id, supplier_name, limit)
                if cached_payloads is not None:
                    return cached_payloads
                try:
                    results, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=models.Filter(must=must),
                        with_payload=True,
                        with_vectors=False,
                        limit=limit,
                    )
                except Exception as retry_exc:
                    if self._is_missing_index_error(retry_exc):
                        logger.warning(
                            "SupplierRelationshipService: payload indexes still missing after retry. Falling back to in-memory filtering.",
                            exc_info=True,
                        )
                        matches = self._fetch_relationship_without_index(
                            supplier_id=supplier_id, supplier_name=supplier_name, limit=limit
                        )
                        if matches:
                            for payload in matches:
                                self._store_fallback_payload(payload)
                        else:
                            self._record_fallback_miss(supplier_id, supplier_name)
                        return matches
                    logger.exception("Failed to load supplier relationship from Qdrant")
                    return []
            else:
                logger.exception("Failed to load supplier relationship from Qdrant")
                return []

        payloads = self._filter_payloads(
            self._extract_payloads(results or []), supplier_id, supplier_name
        )
        if payloads:
            for payload in payloads:
                self._store_fallback_payload(payload)
        else:
            self._record_fallback_miss(supplier_id, supplier_name)
        return payloads

    def fetch_overview(self) -> Optional[Dict[str, Any]]:
        if not self.client or not self.collection_name or not hasattr(self.client, "scroll"):
            return None

        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_type",
                            match=models.MatchValue(value="supplier_relationship_overview"),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=1,
            )
        except Exception:
            logger.exception("Failed to load supplier relationship overview from Qdrant")
            return None

        for point in results or []:
            payload = getattr(point, "payload", None) or {}
            return dict(payload)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_payload_indexes(self) -> None:
        if not self.client or not self.collection_name:
            return

        get_collection = getattr(self.client, "get_collection", None)
        create_index = getattr(self.client, "create_payload_index", None)
        if not callable(get_collection) or not callable(create_index):
            return

        required_indexes = {
            "supplier_id": models.PayloadSchemaType.KEYWORD,
            "supplier_name_normalized": models.PayloadSchemaType.KEYWORD,
        }

        try:
            collection = get_collection(collection_name=self.collection_name)
        except Exception:
            logger.debug(
                "SupplierRelationshipService could not fetch Qdrant collection metadata", exc_info=True
            )
            return

        payload_schema = getattr(collection, "payload_schema", {}) or {}
        existing = set(payload_schema.keys())

        for field, schema in required_indexes.items():
            if field in existing:
                continue
            try:
                create_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
                logger.info(
                    "SupplierRelationshipService created missing Qdrant index for '%s'", field
                )
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code == HTTPStatus.CONFLICT:
                    logger.debug(
                        "SupplierRelationshipService index '%s' already exists", field
                    )
                    continue
                logger.warning(
                    "SupplierRelationshipService failed to create Qdrant index '%s': %s",
                    field,
                    exc,
                )

    def _summarise_flow(self, flow: Mapping[str, Any]) -> tuple[str, List[str]]:
        supplier_id = flow.get("supplier_id")
        supplier_name = str(flow.get("supplier_name") or "").strip()
        label = supplier_name or f"Supplier {supplier_id}" if supplier_id else "Supplier"
        coverage = flow.get("coverage_ratio")
        statements: List[str] = []

        contracts = flow.get("contracts") if isinstance(flow.get("contracts"), Mapping) else {}
        if contracts and contracts.get("count"):
            ids = self._ids_clause(contracts.get("contract_ids"))
            end = contracts.get("latest_end_date")
            suffix = f" ending by {end}" if end else ""
            statements.append(
                f"Company contract records show {int(contracts['count'])} agreement"
                f"{'s' if contracts['count'] != 1 else ''}{ids}{suffix}."
            )

        purchase_orders = flow.get("purchase_orders") if isinstance(flow.get("purchase_orders"), Mapping) else {}
        if purchase_orders and purchase_orders.get("count"):
            ids = self._ids_clause(purchase_orders.get("po_ids"))
            total = purchase_orders.get("total_value_gbp")
            total_clause = f" totalling {total:,.2f} GBP" if isinstance(total, (int, float)) and total else ""
            recent = purchase_orders.get("latest_order_date")
            suffix = f" latest order {recent}" if recent else ""
            statements.append(
                f"Purchase order records link to the supplier identifier across {int(purchase_orders['count'])} order"
                f"{'s' if purchase_orders['count'] != 1 else ''}{ids}{total_clause}{suffix}."
            )

        invoices = flow.get("invoices") if isinstance(flow.get("invoices"), Mapping) else {}
        if invoices and invoices.get("count"):
            ids = self._ids_clause(invoices.get("invoice_ids"))
            total = invoices.get("total_value_gbp")
            total_clause = f" worth {total:,.2f} GBP" if isinstance(total, (int, float)) and total else ""
            recent = invoices.get("latest_invoice_date")
            suffix = f" latest invoice {recent}" if recent else ""
            statements.append(
                f"Accounts payable records connect to the supplier's purchase orders across {int(invoices['count'])} invoice"
                f"{'s' if invoices['count'] != 1 else ''}{ids}{total_clause}{suffix}."
            )

        quotes = flow.get("quotes") if isinstance(flow.get("quotes"), Mapping) else {}
        if quotes and quotes.get("count"):
            ids = self._ids_clause(quotes.get("quote_ids"))
            total = quotes.get("total_value_gbp")
            total_clause = f" valued at {total:,.2f} GBP" if isinstance(total, (int, float)) and total else ""
            statements.append(
                f"Quote exchanges capture {int(quotes['count'])} submission{'s' if quotes['count'] != 1 else ''}{ids}{total_clause}."
            )

        products = flow.get("products") if isinstance(flow.get("products"), Mapping) else {}
        top_items = products.get("top_items") if isinstance(products, Mapping) else None
        if top_items:
            highlights: List[str] = []
            for item in list(top_items)[:5]:
                if not isinstance(item, Mapping):
                    continue
                desc = str(item.get("description") or "").strip()
                spend = item.get("spend_gbp")
                if desc:
                    if isinstance(spend, (int, float)) and spend:
                        highlights.append(f"{desc} ({spend:,.2f} GBP)")
                    else:
                        highlights.append(desc)
            if highlights:
                statements.append(
                    "Top line items span " + ", ".join(highlights) + "."
                )
        unique_items = products.get("unique_items") if isinstance(products, Mapping) else None
        if isinstance(unique_items, (int, float)) and unique_items:
            statements.append(f"Product catalogue covers {int(unique_items)} unique descriptions.")

        intro = f"{label} ({supplier_id}) relationship summary." if supplier_id else f"{label} relationship summary."
        coverage_clause = (
            f" Coverage across contracts, purchase orders, invoices and quotes is {coverage:.2f}."
            if isinstance(coverage, (int, float))
            else ""
        )
        summary = intro + coverage_clause
        if statements:
            summary += " " + " ".join(statements)
        return summary.strip(), statements

    def _build_overview(self, flows: List[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not flows:
            return None

        total_suppliers = 0
        coverage_values: List[float] = []
        totals = {"contracts": 0, "purchase_orders": 0, "invoices": 0, "quotes": 0}
        spend_totals = {"purchase_orders": 0.0, "invoices": 0.0, "quotes": 0.0}
        top_by_coverage: List[Dict[str, Any]] = []
        top_by_spend: List[Dict[str, Any]] = []

        for flow in flows:
            supplier_id = flow.get("supplier_id")
            supplier_name = flow.get("supplier_name")
            total_suppliers += 1

            coverage = flow.get("coverage_ratio")
            if isinstance(coverage, (int, float)):
                coverage_values.append(float(coverage))

            entry_base = {
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
            }

            if isinstance(coverage, (int, float)):
                top_by_coverage.append({**entry_base, "coverage_ratio": float(coverage)})

            spend_value = 0.0
            for key in ("contracts", "purchase_orders", "invoices", "quotes"):
                details = flow.get(key)
                if isinstance(details, Mapping):
                    count = details.get("count")
                    if isinstance(count, (int, float)):
                        totals[key] += int(count)
                if key in ("purchase_orders", "invoices", "quotes") and isinstance(details, Mapping):
                    amount = self._safe_float(details.get("total_value_gbp"))
                    spend_totals[key] += amount
                    spend_value += amount

            if spend_value:
                top_by_spend.append({**entry_base, "total_value_gbp": round(spend_value, 2)})

        top_by_coverage.sort(key=lambda item: item.get("coverage_ratio", 0.0), reverse=True)
        top_by_spend.sort(key=lambda item: item.get("total_value_gbp", 0.0), reverse=True)

        average_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        total_spend = sum(spend_totals.values())

        statements: List[str] = []
        statements.append(
            f"Indexed {total_suppliers} suppliers with average coverage {average_coverage * 100:.1f}% across procurement stages."
        )
        statements.append(
            f"Contracts: {totals['contracts']}, purchase orders: {totals['purchase_orders']}, invoices: {totals['invoices']}, quotes: {totals['quotes']}."
        )
        if total_spend:
            statements.append(
                f"Recorded cumulative spend of {total_spend:,.2f} GBP across purchase orders, invoices and quotes."
            )

        def _format_top(entries: List[Dict[str, Any]], field: str, suffix: str) -> Optional[str]:
            if not entries:
                return None
            descriptions: List[str] = []
            for item in entries[:3]:
                supplier = item.get("supplier_name") or item.get("supplier_id")
                if not supplier:
                    continue
                value = item.get(field)
                if isinstance(value, (int, float)):
                    if field == "coverage_ratio":
                        descriptions.append(f"{supplier} ({value * 100:.1f}% {suffix})")
                    else:
                        descriptions.append(f"{supplier} ({value:,.2f} {suffix})")
                else:
                    descriptions.append(str(supplier))
            if descriptions:
                joined = ", ".join(descriptions)
                return joined
            return None

        coverage_top = _format_top(top_by_coverage, "coverage_ratio", "coverage")
        if coverage_top:
            statements.append(f"Highest coverage suppliers: {coverage_top}.")

        spend_top = _format_top(top_by_spend, "total_value_gbp", "GBP")
        if spend_top:
            statements.append(f"Top spend suppliers: {spend_top}.")

        summary_text = " ".join(statements)

        metadata: Dict[str, Any] = {
            "record_id": "supplier_relationship::overview",
            "document_type": "supplier_relationship_overview",
            "total_suppliers": total_suppliers,
            "average_coverage_ratio": round(average_coverage, 4),
            "total_contracts": totals["contracts"],
            "total_purchase_orders": totals["purchase_orders"],
            "total_invoices": totals["invoices"],
            "total_quotes": totals["quotes"],
            "total_spend_gbp": round(total_spend, 2),
            "spend_breakdown": {k: round(v, 2) for k, v in spend_totals.items()},
            "top_suppliers_by_coverage": top_by_coverage[:5],
            "top_suppliers_by_spend": top_by_spend[:5],
            "relationship_statements": statements,
        }

        return {"text": summary_text, "metadata": metadata}

    @staticmethod
    def _record_id(supplier_id: Optional[str], supplier_name: Optional[str]) -> str:
        if supplier_id:
            return f"supplier_relationship::{supplier_id}"
        if supplier_name:
            cleaned = supplier_name.strip().replace(" ", "_")
            return f"supplier_relationship::{cleaned.lower()}"
        return "supplier_relationship::unknown"

    @staticmethod
    def _normalise_key(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @staticmethod
    def _ids_clause(ids: Optional[Iterable[Any]]) -> str:
        if not ids:
            return ""
        cleaned: List[str] = []
        for raw in ids:
            token = str(raw).strip()
            if token and token not in cleaned:
                cleaned.append(token)
        if not cleaned:
            return ""
        sample = cleaned[:5]
        joined = ", ".join(sample)
        if len(cleaned) > len(sample):
            joined = f"{joined}, ..."
        return f" including {joined}"

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if isinstance(value, str):
                candidate = value.replace(",", "")
                return float(candidate)
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _fetch_relationship_without_index(
        self,
        supplier_id: Optional[str],
        supplier_name: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.client or not self.collection_name:
            return []

        page_size = max(limit, 50)
        max_points_to_scan = max(2000, page_size * 40)
        next_offset = None
        scanned = 0
        matches: List[Dict[str, Any]] = []

        while True:
            try:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    with_payload=True,
                    with_vectors=False,
                    limit=page_size,
                    offset=next_offset,
                )
            except Exception:
                logger.exception(
                    "SupplierRelationshipService fallback scroll failed while loading supplier relationship"
                )
                break

                if not results:
                    break

            scanned += len(results)
            for payload in self._extract_payloads(results):
                document_type = payload.get("document_type")
                if document_type and document_type != "supplier_relationship":
                    continue

                filtered = self._filter_payloads([payload], supplier_id, supplier_name)
                if not filtered:
                    continue

                match = filtered[0]
                self._store_fallback_payload(match)
                matches.append(match)
                if len(matches) >= limit:
                    return matches

            cache = cache_builder
            self._fallback_cache = cache

        matches: List[Dict[str, Any]] = []
        if target_id:
            matches.extend(cache.get("id", {}).get(target_id, []))
        if target_name:
            matches.extend(cache.get("name", {}).get(target_name, []))

        if not matches:
            return []

        deduped: List[Dict[str, Any]] = []
        seen: set[Any] = set()
        for payload in matches:
            record_id = payload.get("record_id") or (
                str(payload.get("supplier_id") or ""),
                payload.get("supplier_name_normalized") or self._normalise_key(payload.get("supplier_name")),
            )
            if record_id in seen:
                continue
            seen.add(record_id)
            deduped.append(payload)

            if scanned >= max_points_to_scan:
                logger.warning(
                    "SupplierRelationshipService fallback scan aborted after inspecting %s records without finding enough"
                    " matches", scanned
                )
                break

            if scanned >= max_points_to_scan:
                logger.warning(
                    "SupplierRelationshipService fallback scan aborted after inspecting %s records without finding enough"
                    " matches", scanned
                )
                break

        if not matches:
            self._record_fallback_miss(supplier_id, supplier_name)
        return matches

    @staticmethod
    def _is_missing_index_error(exc: Exception) -> bool:
        message = str(getattr(exc, "message", "") or exc)
        message = message.lower()
        return "index required but not found" in message

    @staticmethod
    def _extract_payloads(points: Sequence[Any]) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            payloads.append(dict(payload))
        return payloads

    def _filter_payloads(
        self,
        payloads: Sequence[Mapping[str, Any]],
        supplier_id: Optional[str],
        supplier_name: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not payloads:
            return []

        target_id = str(supplier_id).strip() if supplier_id else None
        target_name = self._normalise_key(supplier_name) if supplier_name else None

        filtered: List[Dict[str, Any]] = []
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            sid_value = payload.get("supplier_id")
            if target_id and str(sid_value).strip() != target_id:
                continue
            if target_name:
                payload_name = payload.get("supplier_name_normalized") or self._normalise_key(
                    payload.get("supplier_name")
                )
                if payload_name != target_name:
                    continue
            filtered.append(dict(payload))
        return filtered

    def _lookup_cached_relationship(
        self,
        supplier_id: Optional[str],
        supplier_name: Optional[str],
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        keys = self._cache_lookup_keys(supplier_id, supplier_name)
        if not keys:
            return None

        hits: List[Dict[str, Any]] = []
        seen_records: set[str] = set()
        for key in keys:
            if key in self._fallback_cache:
                payload = self._fallback_cache[key]
                self._fallback_cache.move_to_end(key)
                record = payload.get("record_id") or self._record_id(
                    payload.get("supplier_id"), payload.get("supplier_name")
                )
                if record not in seen_records:
                    hits.append(dict(payload))
                    seen_records.add(record)

        if hits:
            return hits[: max(1, limit)]

        for key in keys:
            if key in self._fallback_negative_cache:
                self._fallback_negative_cache.move_to_end(key)
                return []

        return None

    def _store_fallback_payload(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            return
        cache_payload = dict(payload)
        keys = []
        supplier_id = cache_payload.get("supplier_id")
        if supplier_id is not None:
            key = f"id::{str(supplier_id).strip()}"
            if key:
                keys.append(key)
        normalised = cache_payload.get("supplier_name_normalized") or self._normalise_key(
            cache_payload.get("supplier_name")
        )
        if normalised:
            keys.append(f"name::{normalised}")

        if not keys:
            return

        for key in keys:
            self._fallback_cache.pop(key, None)
            self._fallback_cache[key] = cache_payload
        self._trim_cache(self._fallback_cache, self._fallback_cache_limit)

        for key in keys:
            if key in self._fallback_negative_cache:
                self._fallback_negative_cache.pop(key, None)

    def _record_fallback_miss(
        self, supplier_id: Optional[str], supplier_name: Optional[str]
    ) -> None:
        keys = self._cache_lookup_keys(supplier_id, supplier_name)
        if not keys:
            return
        for key in keys:
            self._fallback_negative_cache.pop(key, None)
            self._fallback_negative_cache[key] = None
        self._trim_cache(self._fallback_negative_cache, self._fallback_cache_limit)

    def _cache_lookup_keys(
        self, supplier_id: Optional[str], supplier_name: Optional[str]
    ) -> List[str]:
        keys: List[str] = []
        if supplier_id:
            sid = str(supplier_id).strip()
            if sid:
                keys.append(f"id::{sid}")
        normalised = self._normalise_key(supplier_name) if supplier_name else None
        if normalised:
            keys.append(f"name::{normalised}")
        return keys

    @staticmethod
    def _trim_cache(cache: "OrderedDict[str, Any]", limit: int) -> None:
        if limit <= 0:
            cache.clear()
            return
        while len(cache) > limit:
            cache.popitem(last=False)


class SupplierRelationshipScheduler:
    NON_BUSINESS_START: time = time(19, 0)
    NON_BUSINESS_END: time = time(7, 0)

    def __init__(self, agent_nick) -> None:
        self.agent_nick = agent_nick
        self.service = SupplierRelationshipService(agent_nick)

    def schedule_daily_refresh(self, scheduled_for: Optional[datetime] = None) -> None:
        self._ensure_schedule_table()
        target = (scheduled_for or self._next_non_business_window()).astimezone(timezone.utc)
        target_date = target.date()

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT status
                        FROM proc.supplier_relationship_refresh
                        WHERE scheduled_for::date = %s
                        ORDER BY scheduled_for DESC
                        LIMIT 1
                        """,
                        (target_date,),
                    )
                    row = cur.fetchone()
                    if row and row[0] == "completed":
                        target = self._next_non_business_window(target + timedelta(days=1))
                        target_date = target.date()
                    cur.execute(
                        """
                        SELECT 1
                        FROM proc.supplier_relationship_refresh
                        WHERE scheduled_for::date = %s
                          AND status IN ('pending', 'in_progress')
                        LIMIT 1
                        """,
                        (target_date,),
                    )
                    if cur.fetchone():
                        return
                    cur.execute(
                        """
                        INSERT INTO proc.supplier_relationship_refresh (scheduled_for)
                        VALUES (%s)
                        """,
                        (target,),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to schedule supplier relationship refresh job")

    def dispatch_due_jobs(
        self,
        moment: Optional[datetime] = None,
        *,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not force and not self._is_non_business_hour(moment):
            logger.info("Skipping supplier relationship refresh; within business hours")
            return []

        self._ensure_schedule_table()
        jobs: List[Dict[str, Any]] = []
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT job_id, scheduled_for
                        FROM proc.supplier_relationship_refresh
                        WHERE status = 'pending' AND scheduled_for <= %s
                        ORDER BY scheduled_for, job_id
                        """,
                        (moment,),
                    )
                    rows = cur.fetchall() or []
                for job_id, scheduled_for in rows:
                    jobs.append({"job_id": job_id, "scheduled_for": scheduled_for})
        except Exception:
            logger.exception("Failed to load supplier relationship refresh jobs")
            return []

        for job in jobs:
            job_id = job["job_id"]
            try:
                self._set_status(job_id, "in_progress", "Refresh started")
                summary = self._run_refresh_job(job_id)
                overview = summary.get("overview") or {}
                supplier_count = overview.get("total_suppliers")
                average_cov = overview.get("average_coverage_ratio")
                message = "Indexed supplier relationships"
                if supplier_count is not None:
                    message += f" for {supplier_count} suppliers"
                if isinstance(average_cov, (int, float)):
                    message += f" (avg coverage {average_cov * 100:.1f}%)"
                self._set_status(job_id, "completed", message.strip())
                job.update(summary)
            except Exception as exc:
                logger.exception("Supplier relationship refresh job %s failed", job_id)
                self._set_status(job_id, "failed", str(exc))
        return jobs

    def run_immediate_refresh(self) -> Dict[str, Any]:
        summary = self._run_refresh_job(job_id=None)
        return summary

    def _run_refresh_job(self, job_id: Optional[int]) -> Dict[str, Any]:
        from engines.query_engine import QueryEngine

        engine = QueryEngine(self.agent_nick)
        df = engine.train_procurement_context()
        overview = self.service.fetch_overview()
        row_count = int(df.shape[0]) if hasattr(df, "shape") else 0
        return {
            "job_id": job_id,
            "rows_indexed": row_count,
            "overview": overview,
        }

    def _ensure_schedule_table(self) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.supplier_relationship_refresh (
                            job_id BIGSERIAL PRIMARY KEY,
                            scheduled_for TIMESTAMPTZ NOT NULL,
                            status TEXT NOT NULL DEFAULT 'pending',
                            result_summary TEXT,
                            created_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_on TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS supplier_relationship_refresh_pending_idx
                            ON proc.supplier_relationship_refresh (status, scheduled_for)
                        """
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to ensure supplier relationship refresh table")

    def _is_non_business_hour(self, moment: datetime) -> bool:
        ts = moment.astimezone(timezone.utc)
        current = ts.time()
        start = self.NON_BUSINESS_START
        end = self.NON_BUSINESS_END
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _next_non_business_window(self, moment: Optional[datetime] = None) -> datetime:
        moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self._is_non_business_hour(moment):
            return moment
        start = moment.replace(
            hour=self.NON_BUSINESS_START.hour,
            minute=self.NON_BUSINESS_START.minute,
            second=0,
            microsecond=0,
        )
        if moment.time() > self.NON_BUSINESS_START:
            start = start + timedelta(days=1)
        return start

    def _set_status(self, job_id: int, status: str, message: Optional[str]) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.supplier_relationship_refresh
                        SET status = %s,
                            result_summary = %s,
                            updated_on = NOW()
                        WHERE job_id = %s
                        """,
                        (status, message, job_id),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to update supplier relationship job status")

