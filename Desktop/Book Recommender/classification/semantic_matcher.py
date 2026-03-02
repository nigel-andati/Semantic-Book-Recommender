"""
Semantic vocabulary matcher.
Uses embeddings to map text to concepts by meaning, not just keywords.
Catches "feeling down" -> sad, "postcolonial" -> colonialism, etc.
Only active when real embeddings (OpenAI or sentence-transformers) are available.
"""

from typing import Callable, List, Optional

from classification.labels import ATMOSPHERE_LABELS, EMOTION_LABELS, SEMANTIC_VOCAB, THEME_LABELS


def _get_concept_descriptors() -> List[tuple[str, str]]:
    """
    Build (canonical_label, descriptor_text) for embedding.
    Only includes concepts with SEMANTIC_VOCAB entries (phrase-level expansions).
    Descriptor = label + full synonym list for rich semantic match.
    """
    result = []
    valid_labels = set(EMOTION_LABELS + THEME_LABELS + ATMOSPHERE_LABELS)

    for label, synonyms in SEMANTIC_VOCAB.items():
        if label not in valid_labels:
            continue
        # Use full phrase list for better matching ("a sad story about colonialism")
        descriptor = " ".join([label] + synonyms[:8])
        result.append((label, descriptor))

    return result


class SemanticMatcher:
    """
    Maps text to emotional/thematic concepts via embedding similarity.
    Use when embeddings (OpenAI or sentence-transformers) are available.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        top_k: int = 5,
        min_score: float = 0.3,
    ):
        self.embed_fn = embed_fn
        self.top_k = top_k
        self.min_score = min_score
        self._concepts: Optional[List[tuple[str, str, List[float]]]] = None

    def _ensure_embedded(self):
        """Lazy-embed concept descriptors."""
        if self._concepts is not None:
            return
        import numpy as np

        concepts = []
        for label, descriptor in _get_concept_descriptors():
            try:
                vec = self.embed_fn(descriptor)
                concepts.append((label, descriptor, np.array(vec, dtype=np.float32)))
            except Exception:
                pass
        self._concepts = concepts

    def match(self, text: str) -> List[dict]:
        """
        Return top matching concepts with scores.
        Each item: {"label": str, "score": float}
        """
        if not text or len(text.strip()) < 3:
            return []

        try:
            import numpy as np

            self._ensure_embedded()
            if not self._concepts:
                return []

            text_vec = np.array(self.embed_fn(text.strip()), dtype=np.float32)
            text_norm = text_vec / (np.linalg.norm(text_vec) + 1e-9)

            scores = []
            for label, _, concept_vec in self._concepts:
                sim = float(np.dot(text_norm, concept_vec))
                if sim >= self.min_score:
                    scores.append((label, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            return [{"label": l, "score": s} for l, s in scores[: self.top_k]]

        except Exception:
            return []
