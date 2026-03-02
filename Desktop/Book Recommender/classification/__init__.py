"""Zero-shot genre classification and sentiment/emotion analysis."""

from .analyzer import BookAnalyzer, analyze_books

__all__ = ["BookAnalyzer", "analyze_books"]
