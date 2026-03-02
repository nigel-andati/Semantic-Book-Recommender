"""
Classification: Multi-category zero-shot with precision-focused logic.
Prioritizes precise thematic alignment over recall. Uses phrase-level hypothesis
templates and semantic vocabulary matching when embeddings are available.
"""

from typing import Callable, Dict, List, Optional

from transformers import pipeline

from classification.labels import (
    ATMOSPHERE_LABELS,
    EMOTION_LABELS,
    GENRE_LABELS,
    THEME_LABELS,
)
from config import ZERO_SHOT_MODEL

# Sentiment analysis model
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Precision over recall: only return labels above this score
MIN_SCORE_THRESHOLD = 0.42
# Max labels per category (fewer = higher precision)
MAX_LABELS_PER_CATEGORY = 3
# Text length for classification (longer = more context, better phrase understanding)
CLASSIFICATION_TEXT_LENGTH = 768


class BookAnalyzer:
    """
    Multi-category zero-shot classification with precision focus.
    - Higher score thresholds to reduce false positives
    - Phrase-level hypothesis templates for full-sentence understanding
    - Semantic matcher for indirect phrasing when embeddings available
    - Description-weighted: uses full text, not metadata tags
    """

    def __init__(
        self,
        zero_shot_model: str = ZERO_SHOT_MODEL,
        sentiment_model: str = SENTIMENT_MODEL,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        use_semantic_matcher: bool = True,
        min_score: float = MIN_SCORE_THRESHOLD,
        max_labels: int = MAX_LABELS_PER_CATEGORY,
    ):
        self._zero_shot = None
        self._zero_shot_model = zero_shot_model
        self._sentiment = None
        self._sentiment_model = sentiment_model
        self.embed_fn = embed_fn
        self.use_semantic_matcher = use_semantic_matcher and embed_fn is not None
        self._semantic_matcher = None
        self.min_score = min_score
        self.max_labels = max_labels

    @property
    def zero_shot_classifier(self):
        if self._zero_shot is None:
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model=self._zero_shot_model,
                device=-1,
            )
        return self._zero_shot

    @property
    def sentiment_analyzer(self):
        if self._sentiment is None:
            self._sentiment = pipeline(
                "sentiment-analysis",
                model=self._sentiment_model,
                device=-1,  # CPU
                return_all_scores=True,
            )
        return self._sentiment

    @property
    def semantic_matcher(self):
        if self._semantic_matcher is None and self.embed_fn is not None:
            from classification.semantic_matcher import SemanticMatcher

            self._semantic_matcher = SemanticMatcher(
                embed_fn=self.embed_fn,
                top_k=5,
                min_score=0.40,  # Higher for precision
            )
        return self._semantic_matcher

    def _classify(
        self,
        text: str,
        candidates: List[str],
        top_k: int,
        hypothesis_template: str,
    ) -> List[Dict]:
        """
        Zero-shot classification with precision filtering.
        Only returns labels above min_score threshold.
        """
        text = str(text).strip()
        if len(text) < 15:
            return []

        try:
            # Use longer context for phrase-level understanding
            truncated = text[:CLASSIFICATION_TEXT_LENGTH]
            result = self.zero_shot_classifier(
                truncated,
                candidate_labels=candidates,
                multi_label=True,
                hypothesis_template=hypothesis_template,
            )
            # Filter by threshold: reduce false positives
            filtered = [
                {"label": l, "score": float(s)}
                for l, s in zip(result["labels"], result["scores"])
                if float(s) >= self.min_score
            ]
            return filtered[:top_k]
        except Exception:
            return []

    def _merge_semantic(
        self,
        zero_shot_results: List[Dict],
        semantic_results: List[Dict],
        top_k: int,
        weight_zero_shot: float = 0.75,  # Slightly favor zero-shot for precision
        weight_semantic: float = 0.25,
    ) -> List[Dict]:
        """
        Merge zero-shot and semantic. Zero-shot primary; semantic adds
        phrase-level matches. Only include labels above threshold.
        """
        if not semantic_results:
            return zero_shot_results[:top_k]

        combined: Dict[str, float] = {}
        for r in zero_shot_results:
            combined[r["label"]] = r["score"] * weight_zero_shot

        for r in semantic_results:
            label = r["label"]
            s = r["score"] * weight_semantic
            combined[label] = combined.get(label, 0) + s

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        # Apply threshold to merged scores
        filtered = [(l, s) for l, s in sorted_items if s >= self.min_score * 0.8]
        return [{"label": l, "score": s} for l, s in filtered[:top_k]]

    def classify_emotion(self, text: str, top_k: int = None) -> List[Dict]:
        """Emotional tone. Phrase-level template for full-sentence understanding."""
        k = top_k or self.max_labels
        zs = self._classify(
            text,
            EMOTION_LABELS,
            top_k=k * 2,  # Get more, then filter
            hypothesis_template="The overall emotional tone of this narrative is predominantly {}.",
        )
        if self.use_semantic_matcher and self.semantic_matcher:
            sem = self.semantic_matcher.match(text)
            emotion_set = set(EMOTION_LABELS)
            sem_filtered = [r for r in sem if r["label"] in emotion_set]
            return self._merge_semantic(zs, sem_filtered, k)
        return zs[:k]

    def classify_theme(self, text: str, top_k: int = None) -> List[Dict]:
        """Thematic classification. Encourages full-phrase interpretation."""
        k = top_k or self.max_labels
        zs = self._classify(
            text,
            THEME_LABELS,
            top_k=k * 2,
            hypothesis_template="This text is centrally about {}.",
        )
        if self.use_semantic_matcher and self.semantic_matcher:
            sem = self.semantic_matcher.match(text)
            theme_set = set(THEME_LABELS)
            sem_filtered = [r for r in sem if r["label"] in theme_set]
            return self._merge_semantic(zs, sem_filtered, k)
        return zs[:k]

    def classify_genre(self, text: str, top_k: int = None) -> List[Dict]:
        """Genre classification."""
        k = top_k or self.max_labels
        return self._classify(
            text,
            GENRE_LABELS,
            top_k=k,
            hypothesis_template="This text is primarily {}.",
        )

    def classify_atmosphere(self, text: str, top_k: int = None) -> List[Dict]:
        """Atmosphere/vibe. Phrase-level for 'fast-paced thriller' etc."""
        k = top_k or self.max_labels
        zs = self._classify(
            text,
            ATMOSPHERE_LABELS,
            top_k=k * 2,
            hypothesis_template="The narrative atmosphere and reading experience is {}.",
        )
        if self.use_semantic_matcher and self.semantic_matcher:
            sem = self.semantic_matcher.match(text)
            atm_set = set(ATMOSPHERE_LABELS)
            sem_filtered = [r for r in sem if r["label"] in atm_set]
            return self._merge_semantic(zs, sem_filtered, k)
        return zs[:k]

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Sentiment analysis using RoBERTa model.
        Returns sentiment scores for negative, neutral, positive.
        """
        text = str(text).strip()
        if len(text) < 10:  # Need minimum text for reliable analysis
            return {"negative": 0.0, "neutral": 0.5, "positive": 0.5}

        try:
            # Truncate to reasonable length for sentiment analysis
            truncated = text[:512]  # RoBERTa has 512 token limit
            results = self.sentiment_analyzer(truncated)

            # Convert to simple dict format
            sentiment_scores = {}
            for result in results[0]:  # results is list of dicts
                label = result["label"].lower()
                score = float(result["score"])
                if "negative" in label or "neg" in label:
                    sentiment_scores["negative"] = score
                elif "positive" in label or "pos" in label:
                    sentiment_scores["positive"] = score
                elif "neutral" in label:
                    sentiment_scores["neutral"] = score

            # Ensure all three categories are present
            sentiment_scores.setdefault("negative", 0.0)
            sentiment_scores.setdefault("neutral", 0.0)
            sentiment_scores.setdefault("positive", 0.0)

            return sentiment_scores

        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return {"negative": 0.0, "neutral": 0.5, "positive": 0.5}

    def analyze_single(self, text: str) -> Dict:
        """
        Full analysis: emotion, theme, genre, atmosphere, and sentiment.
        Uses description text (page_content/combined_text) - not metadata.
        Precision-focused: fewer, higher-confidence labels.
        """
        return {
            "emotions": self.classify_emotion(text),
            "themes": self.classify_theme(text),
            "genres": self.classify_genre(text),
            "atmosphere": self.classify_atmosphere(text),
            "sentiment": self.analyze_sentiment(text),
        }


def analyze_books(
    documents: List[Dict],
    analyzer: Optional[BookAnalyzer] = None,
    text_key: str = "combined_text",
) -> List[Dict]:
    """Attach labels to documents. Prefers combined_text over metadata."""
    analyzer = analyzer or BookAnalyzer()
    results = []
    for doc in documents:
        # Weight description: use page_content/combined_text, not metadata tags
        text = doc.get(text_key) or doc.get("page_content", "")
        analysis = analyzer.analyze_single(text)
        results.append({**doc, **analysis})
    return results
