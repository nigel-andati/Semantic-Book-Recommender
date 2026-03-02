"""
Data Pipeline: Load, clean, and preprocess the book dataset.
Handles missing values, duplicates, creates combined text for embeddings,
and performs zero-shot classification and sentiment analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import kagglehub
import json

from config import DATA_DIR, KAGGLE_DATASET


def _find_data_file(path: Path) -> Optional[Path]:
    """Find CSV or JSON file in the downloaded dataset directory."""
    for ext in ["*.csv", "*.json"]:
        files = sorted(path.rglob(ext))
        if files:
            return files[0]
    return None


def load_raw_data() -> pd.DataFrame:
    """
    Download dataset via kagglehub and load into DataFrame.
    Supports CSV and JSON formats.
    """
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    data_path = Path(path)
    
    data_file = _find_data_file(data_path)
    if data_file is None:
        raise FileNotFoundError(
            f"No CSV or JSON file found in {path}. "
            "Check the dataset structure on Kaggle."
        )
    
    if data_file.suffix == ".csv":
        df = pd.read_csv(data_file)
    else:
        df = pd.read_json(data_file)
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        elif isinstance(df, list):
            df = pd.DataFrame(df)
    
    return df


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df = df.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def _map_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common variations of column names to standard names.
    Handles datasets with different naming conventions.
    """
    column_mapping = {
        "book_title": "title",
        "booktitle": "title",
        "name": "title",
        "authors": "author",
        "author_name": "author",
        "writer": "author",
        "desc": "description",
        "summary": "description",
        "plot": "description",
        "synopsis": "description",
        "categories": "genre",
        "genres": "genre",
        "category": "genre",
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    # If author column contains lists (e.g. ["Author A"]), flatten to string
    if "author" in df.columns:
        df["author"] = df["author"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    # Fallback: infer description from first long text column if missing
    if "description" not in df.columns:
        for col in df.columns:
            if col in ("title", "author", "genre"):
                continue
            sample = df[col].dropna().astype(str)
            if len(sample) > 0 and sample.str.len().median() > 50:
                df["description"] = df[col].fillna("").astype(str)
                break
        if "description" not in df.columns:
            df["description"] = ""

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Remove duplicates
    - Handle missing values
    - Drop rows without essential fields (title or description)
    """
    df = df.copy()
    # Sort for deterministic duplicate removal (keep first occurrence)
    sort_cols = [c for c in ["title", "author", "description"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Identify text columns for combined field
    text_cols = [c for c in df.columns if c in ["title", "author", "description", "genre"]]
    
    # Fill missing descriptions with empty string (we'll filter later)
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)
    else:
        df["description"] = ""
    
    if "title" in df.columns:
        df["title"] = df["title"].fillna("Unknown").astype(str)
    else:
        df["title"] = "Unknown"
    
    if "author" in df.columns:
        df["author"] = df["author"].fillna("Unknown").astype(str)
    else:
        df["author"] = "Unknown"
    
    if "genre" in df.columns:
        df["genre"] = df["genre"].fillna("").astype(str)
    else:
        df["genre"] = ""
    
    # Drop rows with empty or very short descriptions (not useful for semantic search)
    df = df[df["description"].str.strip().str.len() >= 20]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined text field for embeddings.
    Format: "Title: X | Author: Y | Genre: Z | Description: ..."
    This gives the embedding model rich context for semantic matching.
    """
    df = df.copy()

    parts = []
    if "title" in df.columns:
        parts.append("Title: " + df["title"].astype(str))
    if "author" in df.columns:
        parts.append("Author: " + df["author"].astype(str))
    if "genre" in df.columns and df["genre"].notna().any():
        parts.append("Genre: " + df["genre"].astype(str))
    if "description" in df.columns:
        parts.append("Description: " + df["description"].astype(str))

    # Combine row-wise (each part is a Series)
    if parts:
        df["combined_text"] = parts[0]
        for p in parts[1:]:
            df["combined_text"] = df["combined_text"] + " | " + p
    else:
        df["combined_text"] = df.get("description", pd.Series([""] * len(df)))

    return df


def load_and_clean_data(
    use_cache: bool = True,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main entry point: load, clean, and preprocess the dataset.
    Optionally cache the cleaned data to avoid re-downloading.
    """
    cache_path = cache_path or (DATA_DIR / "cleaned_books.parquet")
    
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    
    df = load_raw_data()
    df = _normalize_column_names(df)
    df = _map_common_columns(df)
    df = clean_data(df)
    df = create_combined_text(df)
    
    # Cache for next time
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    
    return df


def classify_books_batch(
    books_df: pd.DataFrame,
    batch_size: int = 50,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Perform zero-shot classification and sentiment analysis on all books.
    Processes in batches to handle large datasets efficiently.

    Args:
        books_df: DataFrame with book data
        batch_size: Number of books to process at once
        use_cache: Whether to cache results

    Returns:
        DataFrame with classification columns added
    """
    from classification import BookAnalyzer

    cache_path = DATA_DIR / "classified_books_cache.parquet"

    # Check cache first
    if use_cache and cache_path.exists():
        print(f"Loading cached classifications from {cache_path}")
        try:
            cached_df = pd.read_parquet(cache_path)
            # Verify cache has the expected columns
            expected_cols = ["emotions", "themes", "genres", "atmosphere", "sentiment"]
            if all(col in cached_df.columns for col in expected_cols):
                print(f"✅ Loaded {len(cached_df)} classified books from cache")
                return cached_df
        except Exception as e:
            print(f"Cache load failed: {e}, reprocessing...")

    print(f"🔍 Classifying {len(books_df)} books in batches of {batch_size}...")

    # Initialize analyzer (no embeddings needed for zero-shot)
    analyzer = BookAnalyzer(use_semantic_matcher=False)

    # Convert DataFrame to list of dicts for processing
    books_data = books_df.to_dict('records')
    processed_books = []

    for i in range(0, len(books_data), batch_size):
        batch = books_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(books_data) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches}...")

        # Process each book in the batch
        for book in batch:
            text = book.get('combined_text', '')
            if not text or len(text.strip()) < 20:
                # Skip books with insufficient text
                book.update({
                    "emotions": [],
                    "themes": [],
                    "genres": [],
                    "atmosphere": [],
                    "sentiment": {"negative": 0.0, "neutral": 0.5, "positive": 0.5}
                })
            else:
                # Perform full analysis
                analysis = analyzer.analyze_single(text)
                book.update(analysis)

        processed_books.extend(batch)

        # Progress indicator
        processed_count = min(i + batch_size, len(books_data))
        print(f"   Completed {processed_count}/{len(books_data)} books")

    # Convert back to DataFrame
    result_df = pd.DataFrame(processed_books)

    # Convert classification results to JSON strings for storage
    classification_cols = ["emotions", "themes", "genres", "atmosphere", "sentiment"]
    for col in classification_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    # Cache results
    if use_cache:
        print(f"💾 Caching classified books to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(cache_path, index=False)

    print(f"✅ Classification complete! Processed {len(result_df)} books")
    return result_df


def load_classified_books(use_cache: bool = True) -> pd.DataFrame:
    """
    Load books with classifications (emotion, theme, genre, atmosphere, sentiment).

    Args:
        use_cache: Whether to use cached processed data

    Returns:
        DataFrame with all book data and classifications
    """
    # First get the cleaned book data
    books_df = load_and_clean_data(use_cache=use_cache)

    # Then add classifications
    classified_df = classify_books_batch(books_df, use_cache=use_cache)

    # Parse JSON strings back to Python objects
    classification_cols = ["emotions", "themes", "genres", "atmosphere", "sentiment"]
    for col in classification_cols:
        if col in classified_df.columns:
            classified_df[col] = classified_df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') or x.startswith('{') else x
            )

    return classified_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "classify":
        # Full classification pipeline
        print("🚀 Running full classification pipeline...")
        df = load_classified_books(use_cache=True)
        print(f"✅ Processed {len(df)} books with full classification")

        # Show sample results
        sample = df.head(1)
        if not sample.empty:
            print("\n📖 Sample Classification Results:")
            print(f"Title: {sample['title'].iloc[0]}")
            print(f"Author: {sample['author'].iloc[0]}")
            print(f"Genres: {sample['genres'].iloc[0]}")
            print(f"Emotions: {sample['emotions'].iloc[0]}")
            print(f"Themes: {sample['themes'].iloc[0]}")
            print(f"Atmosphere: {sample['atmosphere'].iloc[0]}")
            print(f"Sentiment: {sample['sentiment'].iloc[0]}")
    else:
        # Quick test - just load and clean
        df = load_and_clean_data(use_cache=False)
        print(f"Loaded {len(df)} books")
        print(df[["title", "author", "combined_text"]].head(2))
        print("\n💡 Run with 'python data/pipeline.py classify' to perform full classification")
