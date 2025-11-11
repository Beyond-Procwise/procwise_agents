from __future__ import annotations

"""Feedback detection and storage utilities for RAG interactions."""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeedbackSentiment(Enum):
    """Classification of user feedback."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    NEUTRAL = "neutral"
    CLARIFICATION = "clarification"


class FeedbackService:
    """Detect, store, and process user feedback for continuous learning."""

    POSITIVE_PATTERNS = [
        r"\b(thank(s| you))\b",
        r"\b(perfect|excellent|exactly|spot on|nailed it)\b",
        r"\b(appreciate(s)?|appreciated)\b",
        r"\b(this is (just )?what i (wanted|needed))\b",
        r"\b(that'?s? (right|correct))\b",
        r"👍|✓|✔",
    ]

    NEGATIVE_PATTERNS = [
        r"\b(wrong|incorrect|inaccurate|unacceptable|not what i (wanted|expected|asked))\b",
        r"\b(this (doesn't|does not) (help|work|make sense))\b",
        r"\b(unhelpful|useless|bad answer)\b",
        r"\b(missing|incomplete|confusing)\b",
        r"\b(no,? that'?s? wrong)\b",
        r"👎|✗|✘",
    ]

    CORRECTION_PATTERNS = [
        r"\b(actually|correction|fix|should be|meant to)\b",
        r"\b(the (right|correct) (answer|info) is)\b",
        r"\b(it'?s? (really|actually))\b",
    ]

    CLARIFICATION_PATTERNS = [
        r"\b(what do you mean|can you explain|i don't understand|unclear)\b",
    ]

    CONTINUATION_PATTERNS = [
        r"\b(good|great)\b[^.]*\b(but|however)\b",
        r"\b(please|kindly)?\s*(be|get|provide).*(more detail|detailed|detail|elaborate)\b",
        r"\b(tell me more|please elaborate|elaborate more)\b",
        r"\b(what about|how about)\b",
        r"\b(can you add (more )?(detail|details|limits))\b",
        r"\b(be more detailed)\b",
    ]

    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        self._ensure_feedback_table()

    # ------------------------------------------------------------------
    # Feedback Detection
    # ------------------------------------------------------------------

    def detect_feedback(self, message: str) -> Tuple[FeedbackSentiment, float]:
        """Analyze ``message`` for feedback signals."""

        message_lower = message.lower().strip()

        if self.is_continuation_request(message_lower):
            return FeedbackSentiment.NEUTRAL, 0.0

        is_short = len(message_lower.split()) < 15

        positive_score = self._pattern_match_score(message_lower, self.POSITIVE_PATTERNS)
        negative_score = self._pattern_match_score(message_lower, self.NEGATIVE_PATTERNS)
        correction_score = self._pattern_match_score(message_lower, self.CORRECTION_PATTERNS)
        clarification_score = self._pattern_match_score(
            message_lower, self.CLARIFICATION_PATTERNS
        )

        scores = {
            FeedbackSentiment.POSITIVE: positive_score,
            FeedbackSentiment.NEGATIVE: negative_score,
            FeedbackSentiment.CORRECTION: correction_score,
            FeedbackSentiment.CLARIFICATION: clarification_score,
        }

        max_sentiment = max(scores.keys(), key=lambda key: scores[key])
        max_score = scores[max_sentiment]

        if is_short and max_score > 0.3:
            max_score = min(1.0, max_score * 1.5)

        if max_sentiment == FeedbackSentiment.CORRECTION:
            max_sentiment = FeedbackSentiment.NEGATIVE

        if max_sentiment == FeedbackSentiment.CLARIFICATION:
            return FeedbackSentiment.NEUTRAL, 0.0

        threshold = 0.4 if is_short else 0.2

        if max_score < threshold:
            return FeedbackSentiment.NEUTRAL, 0.0

        return max_sentiment, max_score

    def _pattern_match_score(self, text: str, patterns: List[str]) -> float:
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return min(1.0, matches / max(1, len(patterns) * 0.3))

    def is_feedback_message(self, message: str, confidence_threshold: float = 0.4) -> bool:
        if self.is_continuation_request(message):
            return False
        sentiment, confidence = self.detect_feedback(message)
        return sentiment != FeedbackSentiment.NEUTRAL and confidence >= confidence_threshold

    def is_continuation_request(self, message: str) -> bool:
        text = message.lower().strip()
        if not text:
            return False
        if any(re.search(pattern, text) for pattern in self.CONTINUATION_PATTERNS):
            return True
        if "more detail" in text or "tell me more" in text:
            return True
        if ("good" in text or "great" in text) and ("but" in text or "however" in text):
            return True
        return False

    # ------------------------------------------------------------------
    # Feedback Storage
    # ------------------------------------------------------------------

    def _ensure_feedback_table(self) -> None:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE SCHEMA IF NOT EXISTS proc")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS proc.rag_feedback (
                            feedback_id BIGSERIAL PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            session_id TEXT,
                            query TEXT NOT NULL,
                            response TEXT NOT NULL,
                            feedback_message TEXT NOT NULL,
                            sentiment TEXT NOT NULL,
                            confidence FLOAT NOT NULL,
                            retrieved_doc_ids TEXT[],
                            context_metadata JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            processed BOOLEAN DEFAULT FALSE,
                            training_incorporated BOOLEAN DEFAULT FALSE
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_rag_feedback_user_session
                        ON proc.rag_feedback(user_id, session_id)
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_rag_feedback_sentiment
                        ON proc.rag_feedback(sentiment, processed)
                        """
                    )
                conn.commit()  # Ensure changes persist before connection closes
        except Exception:  # pragma: no cover
            logger.exception("Failed to ensure RAG feedback table")

    def store_feedback(
        self,
        *,
        user_id: str,
        session_id: Optional[str],
        query: str,
        response: str,
        feedback_message: str,
        sentiment: FeedbackSentiment,
        confidence: float,
        retrieved_doc_ids: Optional[List[str]] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.rag_feedback
                            (user_id, session_id, query, response, feedback_message,
                             sentiment, confidence, retrieved_doc_ids, context_metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        RETURNING feedback_id
                        """,
                        (
                            user_id,
                            session_id,
                            query,
                            response,
                            feedback_message,
                            sentiment.value,
                            confidence,
                            retrieved_doc_ids or [],
                            json.dumps(context_metadata or {}, default=str),
                        ),
                    )
                    feedback_id = cur.fetchone()[0]
                conn.commit()
                logger.info(
                    "Stored %s feedback (confidence: %.2f) for user %s, feedback_id: %s",
                    sentiment.value,
                    confidence,
                    user_id,
                    feedback_id,
                )
                return feedback_id
        except Exception:  # pragma: no cover
            logger.exception("Failed to store RAG feedback")
            return None

    # ------------------------------------------------------------------
    # Feedback Retrieval for Training
    # ------------------------------------------------------------------

    def get_unprocessed_feedback(
        self,
        limit: int = 100,
        sentiment_filter: Optional[FeedbackSentiment] = None,
    ) -> List[Dict[str, Any]]:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    if sentiment_filter:
                        cur.execute(
                            """
                            SELECT feedback_id, user_id, session_id, query, response,
                                   feedback_message, sentiment, confidence,
                                   retrieved_doc_ids, context_metadata, created_at
                            FROM proc.rag_feedback
                            WHERE processed = FALSE AND sentiment = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                            """,
                            (sentiment_filter.value, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT feedback_id, user_id, session_id, query, response,
                                   feedback_message, sentiment, confidence,
                                   retrieved_doc_ids, context_metadata, created_at
                            FROM proc.rag_feedback
                            WHERE processed = FALSE
                            ORDER BY created_at DESC
                            LIMIT %s
                            """,
                            (limit,),
                        )
                    rows = cur.fetchall()
                    feedback_records: List[Dict[str, Any]] = []
                    for row in rows:
                        feedback_records.append(
                            {
                                "feedback_id": row[0],
                                "user_id": row[1],
                                "session_id": row[2],
                                "query": row[3],
                                "response": row[4],
                                "feedback_message": row[5],
                                "sentiment": row[6],
                                "confidence": row[7],
                                "retrieved_doc_ids": row[8],
                                "context_metadata": row[9],
                                "created_at": row[10].isoformat() if row[10] else None,
                            }
                        )
                    return feedback_records
        except Exception:  # pragma: no cover
            logger.exception("Failed to retrieve unprocessed feedback")
            return []

    def mark_feedback_processed(self, feedback_ids: List[int]) -> None:
        if not feedback_ids:
            return
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.rag_feedback
                        SET processed = TRUE
                        WHERE feedback_id = ANY(%s)
                        """,
                        (feedback_ids,),
                    )
                conn.commit()
                logger.info(
                    "Marked %d feedback records as processed", len(feedback_ids)
                )
        except Exception:  # pragma: no cover
            logger.exception("Failed to mark feedback as processed")

    def get_feedback_statistics(self) -> Dict[str, Any]:
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT sentiment, COUNT(*) AS count,
                               AVG(confidence) AS avg_confidence
                        FROM proc.rag_feedback
                        WHERE created_at > NOW() - INTERVAL '30 days'
                        GROUP BY sentiment
                        """
                    )
                    rows = cur.fetchall()
                    stats = {
                        "total": 0,
                        "by_sentiment": {},
                        "unprocessed": 0,
                    }
                    for sentiment, count, avg_conf in rows:
                        stats["total"] += count
                        stats["by_sentiment"][sentiment] = {
                            "count": count,
                            "avg_confidence": float(avg_conf) if avg_conf else 0.0,
                        }
                    cur.execute(
                        "SELECT COUNT(*) FROM proc.rag_feedback WHERE processed = FALSE"
                    )
                    stats["unprocessed"] = cur.fetchone()[0]
                    return stats
        except Exception:  # pragma: no cover
            logger.exception("Failed to get feedback statistics")
            return {"total": 0, "by_sentiment": {}, "unprocessed": 0}

    # ------------------------------------------------------------------
    # Acknowledgment Generation
    # ------------------------------------------------------------------

    def generate_acknowledgment(
        self, sentiment: FeedbackSentiment, feedback_message: str
    ) -> str:
        if sentiment == FeedbackSentiment.POSITIVE:
            acknowledgments = [
                "I'm glad that helped! Let me know if you need anything else.",
                "Great to hear! Feel free to ask if you have more questions.",
                "Perfect! I'm here if you need further assistance.",
            ]
        elif sentiment == FeedbackSentiment.NEGATIVE:
            acknowledgments = [
                "Understood—I'll tighten that up below.",
                "Got it, let me correct that right away.",
                "Thanks for flagging it; here's an improved answer.",
            ]
        elif sentiment == FeedbackSentiment.CLARIFICATION:
            acknowledgments = [
                "Of course, let me clarify that for you.",
                "Good question—let me break that down more clearly.",
                "I can provide more detail on that.",
            ]
        else:
            acknowledgments = [
                "I see. Let me take another look at this.",
                "Thank you for that context.",
                "Got it. Let me refine that.",
            ]
        import random

        return random.choice(acknowledgments)
