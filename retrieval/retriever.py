"""
Semantic Retrieval: Accept natural language queries, retrieve top-k similar books.
Uses LangChain's retriever interface over ChromaDB with Sentence Transformers embeddings.
Enhanced with classification-based ranking for better relevance.
"""

from typing import List, Optional, Dict, Any, Tuple
import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore

from config import DEFAULT_TOP_K


class BookRetriever(BaseRetriever):
    """
    Enhanced LangChain retriever that combines semantic similarity with
    classification-based ranking for better book recommendations.
    """

    vectorstore: VectorStore
    k: int = DEFAULT_TOP_K
    search_type: str = "similarity"
    classification_data: Optional[Dict[str, Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # ChromaDB uses precomputed embeddings for fast similarity search
        # No live embedding generation needed during queries
        docs = self.vectorstore.similarity_search(query, k=self.k)
        return docs

    def _analyze_query_preferences(self, query: str) -> Dict[str, List[str]]:
        """
        Extract classification preferences from natural language query.
        Returns dict with keys: emotions, themes, genres, atmosphere.
        """
        preferences = {
            "emotions": [],
            "themes": [],
            "genres": [],
            "atmosphere": []
        }

        query_lower = query.lower()

        # Define keyword mappings for different categories
        emotion_keywords = {
            "happy": ["happy", "joy", "cheerful", "uplifting"],
            "sad": ["sad", "sorrowful", "melancholic", "grief"],
            "dark": ["dark", "scary", "horror", "disturbing"],
            "hopeful": ["hopeful", "inspiring", "redemption"],
            "romantic": ["romantic", "love", "passion"],
            "mysterious": ["mysterious", "suspense", "intrigue"]
        }

        genre_keywords = {
            "romance": ["romance", "love story", "romantic"],
            "mystery": ["mystery", "detective", "whodunit"],
            "thriller": ["thriller", "suspense", "action"],
            "fantasy": ["fantasy", "magic", "magical"],
            "sci-fi": ["science fiction", "sci-fi", "space"],
            "horror": ["horror", "scary", "terror"],
            "historical": ["historical", "history", "period"],
            "comedy": ["comedy", "funny", "humor"]
        }

        theme_keywords = {
            "identity": ["identity", "self-discovery", "belonging"],
            "justice": ["justice", "equality", "rights"],
            "trauma": ["trauma", "healing", "survivor"],
            "colonialism": ["colonialism", "empire", "postcolonial"],
            "migration": ["migration", "immigration", "diaspora"]
        }

        atmosphere_keywords = {
            "mysterious": ["mysterious", "atmospheric", "moody"],
            "fast-paced": ["fast-paced", "action", "thrilling"],
            "slow": ["slow", "reflective", "contemplative"],
            "suspenseful": ["suspenseful", "tense", "nail-biting"]
        }

        # Check each category
        for emotion, keywords in emotion_keywords.items():
            if any(kw in query_lower for kw in keywords):
                preferences["emotions"].append(emotion)

        for genre, keywords in genre_keywords.items():
            if any(kw in query_lower for kw in keywords):
                preferences["genres"].append(genre)

        for theme, keywords in theme_keywords.items():
            if any(kw in query_lower for kw in keywords):
                preferences["themes"].append(theme)

        for atmosphere, keywords in atmosphere_keywords.items():
            if any(kw in query_lower for kw in keywords):
                preferences["atmosphere"].append(atmosphere)

        return preferences

    def _calculate_classification_score(
        self,
        book_classifications: Dict[str, Any],
        query_preferences: Dict[str, List[str]]
    ) -> float:
        """
        Calculate how well a book's classifications match query preferences.
        Returns score between 0-1.
        """
        if not book_classifications or not query_preferences:
            return 0.0

        total_score = 0.0
        max_score = 0.0

        # Check each preference category
        for category, preferred_items in query_preferences.items():
            if not preferred_items:
                continue

            max_score += 1.0  # One point per category with preferences

            book_items = book_classifications.get(category, [])

            # Handle different data formats
            if isinstance(book_items, list):
                if book_items and isinstance(book_items[0], dict):
                    # List of dicts with 'label' key
                    book_labels = [item.get('label', '').lower() for item in book_items]
                else:
                    # List of strings
                    book_labels = [str(item).lower() for item in book_items]
            else:
                book_labels = []

            # Calculate match score for this category
            category_score = 0.0
            for preferred in preferred_items:
                preferred_lower = preferred.lower()
                # Check for exact matches or partial matches
                for book_label in book_labels:
                    if preferred_lower in book_label or book_label in preferred_lower:
                        category_score = max(category_score, 0.8)  # Strong match
                        break
                    # Check for related terms
                    if self._are_related_terms(preferred_lower, book_label):
                        category_score = max(category_score, 0.6)  # Related match
                        break

            total_score += category_score

        return total_score / max_score if max_score > 0 else 0.0

    def _are_related_terms(self, term1: str, term2: str) -> bool:
        """Check if two terms are semantically related."""
        # Simple relatedness check - can be enhanced with word embeddings
        related_pairs = [
            ("happy", "joyful"), ("sad", "sorrowful"), ("dark", "grim"),
            ("scary", "horror"), ("love", "romance"), ("mystery", "detective"),
            ("action", "thriller"), ("magic", "fantasy"), ("space", "sci-fi")
        ]

        term1_lower, term2_lower = term1.lower(), term2.lower()
        return any(
            (term1_lower in pair and term2_lower in pair) or
            (term1_lower in pair[1] and term2_lower in pair[0])
            for pair in related_pairs
        )

    def get_enhanced_recommendations(
        self,
        query: str,
        classification_data: Optional[Dict[str, Dict[str, Any]]] = None,
        semantic_weight: float = 0.7,
        classification_weight: float = 0.3
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Enhanced retrieval combining semantic similarity with classification matching.

        Args:
            query: Natural language search query
            classification_data: Dict mapping document IDs to classification data
            semantic_weight: Weight for semantic similarity (0-1)
            classification_weight: Weight for classification matching (0-1)

        Returns:
            List of (document, combined_score, metadata) tuples
        """
        # Get base semantic results with scores
        semantic_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.k * 2  # Get more candidates for reranking
        )

        # Analyze query for classification preferences
        query_preferences = self._analyze_query_preferences(query)

        # Combine semantic and classification scores
        enhanced_results = []
        for doc, semantic_score in semantic_results:
            # Get book classifications if available
            book_classifications = None
            if classification_data:
                # Try to match by document ID or content
                doc_id = getattr(doc, 'id', None) or getattr(doc, 'metadata', {}).get('id')
                if doc_id and doc_id in classification_data:
                    book_classifications = classification_data[doc_id]

            # Calculate classification score
            classification_score = 0.0
            if book_classifications and query_preferences:
                classification_score = self._calculate_classification_score(
                    book_classifications, query_preferences
                )

            # Combine scores
            combined_score = (
                semantic_weight * semantic_score +
                classification_weight * classification_score
            )

            # Prepare metadata
            metadata = {
                "semantic_score": float(semantic_score),
                "classification_score": classification_score,
                "combined_score": combined_score,
                "query_preferences": query_preferences
            }

            if book_classifications:
                metadata["classifications"] = book_classifications

            enhanced_results.append((doc, combined_score, metadata))

        # Sort by combined score and return top-k
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:self.k]

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> List[tuple[Document, float]]:
        """Return documents with similarity scores for ranking/display."""
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.k
        )


def get_retriever(vectorstore, k: int = DEFAULT_TOP_K) -> BookRetriever:
    """Create a BookRetriever from a Chroma vectorstore."""
    return BookRetriever(vectorstore=vectorstore, k=k)
