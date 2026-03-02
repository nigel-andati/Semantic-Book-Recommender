"""Data pipeline for loading, cleaning, and preprocessing book data."""

from .pipeline import load_and_clean_data, create_combined_text

__all__ = ["load_and_clean_data", "create_combined_text"]
