import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import nltk
    from nltk import pos_tag, word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem import WordNetLemmatizer
    from nltk.tree import Tree
    from nltk.chunk import RegexpParser
    from nltk.corpus import wordnet
except Exception:  # pragma: no cover - runtime fallback only
    nltk = None
    pos_tag = None
    word_tokenize = None
    sent_tokenize = None
    stopwords = None
    SentimentIntensityAnalyzer = None
    WordNetLemmatizer = None
    Tree = None
    RegexpParser = None
    wordnet = None

logger = logging.getLogger(__name__)


@dataclass
class NLTKFeatures:
    clean_text: str
    sentences: List[str]
    tokens: List[str]
    lemmas: List[str]
    keywords: List[str]
    key_phrases: List[str]
    sentiment: Dict[str, float]
    available: bool = True


class NLTKProcessor:
    """Utility wrapper that prepares and cleans LLM inputs and outputs using NLTK."""

    _RESOURCE_MAP = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab/english.pickle": "punkt_tab",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "sentiment/vader_lexicon": "vader_lexicon",
    }

    _NP_VP_GRAMMAR = r"""
        NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}
        VP: {<VB.*><NP|PP|RB>*}
    """

    _FILLER_PREFIXES = (
        "thanks for flagging",
        "here's what i can confirm",
        "as an ai",
        "based on",
        "sure, i can help",
        "most definitely",
        "great! here",
        "you're in the right place",
    )

    _TOXICITY_PATTERNS = (
        re.compile(r"\b(?:idiot|stupid|dumb)\b", re.IGNORECASE),
    )

    _PII_PATTERNS = (
        re.compile(r"\b(?:ssn|social security|credit card|card number)\b", re.IGNORECASE),
    )

    def __init__(self) -> None:
        self.available = self._initialise()
        if not self.available:
            self._stop_words: set[str] = set()
            self._lemmatizer = None
            self._chunker = None
            self._sentiment = None
        else:
            self._stop_words = stopwords.words("english")  # type: ignore[arg-type]
            self._lemmatizer = WordNetLemmatizer()
            self._chunker = RegexpParser(self._NP_VP_GRAMMAR)
            try:
                self._sentiment = SentimentIntensityAnalyzer()
            except LookupError:  # pragma: no cover - depends on corpora availability
                logger.warning("VADER lexicon missing; sentiment scoring disabled")
                self._sentiment = None

    def _initialise(self) -> bool:
        if nltk is None or any(
            module is None
            for module in (pos_tag, word_tokenize, sent_tokenize, stopwords, WordNetLemmatizer)
        ):
            logger.warning("NLTK is not available in this environment; continuing without it")
            return False

        for resource_path, download_name in self._RESOURCE_MAP.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                try:
                    nltk.download(download_name, quiet=True)
                except Exception:  # pragma: no cover - download failure handling
                    logger.warning(
                        "Failed to download NLTK resource %s; NLTK features disabled",
                        download_name,
                    )
                    return False
        return True

    def preprocess(self, text: str) -> NLTKFeatures:
        clean = (text or "").strip()
        if not clean or not self.available:
            return NLTKFeatures(
                clean_text=clean,
                sentences=[],
                tokens=[],
                lemmas=[],
                keywords=[],
                key_phrases=[],
                sentiment={},
                available=self.available,
            )

        sentences = sent_tokenize(clean)  # type: ignore[operator]
        raw_tokens: List[str] = []
        for sentence in sentences:
            raw_tokens.extend(word_tokenize(sentence))  # type: ignore[operator]

        lowered_tokens = [token.lower() for token in raw_tokens]
        table = str.maketrans("", "", string.punctuation)
        filtered = [
            token.translate(table)
            for token in lowered_tokens
            if token.translate(table)
        ]
        filtered = [
            token
            for token in filtered
            if token
            and token not in self._stop_words
            and any(ch.isalpha() for ch in token)
        ]

        pos_tags = pos_tag(raw_tokens)  # type: ignore[operator]
        lemmas = [
            self._lemmatize_token(token.lower(), tag)
            for token, tag in pos_tags
            if token.strip()
        ]
        lemma_counts = Counter(lemma for lemma in lemmas if lemma and lemma not in self._stop_words)
        top_keywords = [item for item, _ in lemma_counts.most_common(8)]

        key_phrases = self._extract_phrases(pos_tags)

        sentiment_scores: Dict[str, float] = {}
        if self._sentiment:
            sentiment_scores = self._sentiment.polarity_scores(clean)

        features = NLTKFeatures(
            clean_text=clean,
            sentences=sentences,
            tokens=filtered,
            lemmas=lemmas,
            keywords=top_keywords,
            key_phrases=key_phrases,
            sentiment=sentiment_scores,
        )

        logger.debug(
            "Derived NLTK features: keywords=%s key_phrases=%s sentiment=%s",
            features.keywords,
            features.key_phrases[:5],
            features.sentiment,
        )
        return features

    def postprocess(
        self,
        text: str,
        *,
        sentiment: Optional[Dict[str, float]] = None,
    ) -> str:
        clean = (text or "").strip()
        if not clean:
            return ""

        if not self.available:
            return self._basic_cleanup(clean)

        sentences = sent_tokenize(clean) if clean else []  # type: ignore[operator]
        filtered: List[str] = []
        for sentence in sentences or [clean]:
            lowered = sentence.strip().lower()
            if any(lowered.startswith(prefix) for prefix in self._FILLER_PREFIXES):
                continue
            filtered.append(sentence.strip())

        if not filtered:
            filtered = [clean]

        tone_sentence = None
        compound = (sentiment or {}).get("compound") if sentiment else None
        if compound is not None and compound < -0.2:
            tone_sentence = "I understand this situation may feel frustrating; let me walk through the relevant guidance."
        elif compound is not None and compound > 0.4:
            tone_sentence = "Great news—this aligns well with the current guidance."

        normalised: List[str] = []
        for idx, sentence in enumerate(filtered):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence = self._replace_sensitive_terms(sentence)
            if sentence[-1] not in ".!?":
                sentence = f"{sentence}."
            sentence = sentence[0].upper() + sentence[1:]
            if idx == 0 and tone_sentence:
                normalised.append(tone_sentence)
            normalised.append(sentence)

        result = " ".join(normalised) if normalised else clean
        result = re.sub(r"\s+", " ", result)
        return result.strip()

    def _lemmatize_token(self, token: str, pos_tag_value: str) -> str:
        if not self._lemmatizer:
            return token
        wn_tag = self._map_pos(pos_tag_value)
        return self._lemmatizer.lemmatize(token, wn_tag) if wn_tag else self._lemmatizer.lemmatize(token)

    def _map_pos(self, tag: str) -> Optional[str]:
        if not tag:
            return None
        first = tag[0].upper()
        mapping = {
            "J": wordnet.ADJ if wordnet else None,
            "N": wordnet.NOUN if wordnet else None,
            "V": wordnet.VERB if wordnet else None,
            "R": wordnet.ADV if wordnet else None,
        }
        return mapping.get(first)

    def _extract_phrases(self, tagged_tokens: List[tuple[str, str]]) -> List[str]:
        if not self._chunker or not tagged_tokens:
            return []
        try:
            tree: Tree = self._chunker.parse(tagged_tokens)
        except Exception:  # pragma: no cover - defensive
            return []

        phrases: List[str] = []
        for subtree in tree.subtrees():
            if subtree.label() not in {"NP", "VP"}:
                continue
            phrase_tokens = [token for token, _ in subtree.leaves()]
            phrase = " ".join(token for token in phrase_tokens if token)
            phrase = phrase.strip()
            if phrase and len(phrase.split()) <= 6:
                phrases.append(phrase)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_phrases: List[str] = []
        for phrase in phrases:
            lowered = phrase.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_phrases.append(phrase)
        return unique_phrases[:8]

    def _replace_sensitive_terms(self, sentence: str) -> str:
        for pattern in self._PII_PATTERNS:
            sentence = pattern.sub("[redacted sensitive reference]", sentence)
        for pattern in self._TOXICITY_PATTERNS:
            sentence = pattern.sub("inappropriate language", sentence)
        return sentence

    def _basic_cleanup(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text)
        for prefix in self._FILLER_PREFIXES:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip(",. ")
        return cleaned
