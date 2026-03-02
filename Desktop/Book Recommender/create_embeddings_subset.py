#!/usr/bin/env python3
"""
Create embeddings for the 7k books dataset using Sentence Transformers.
Output: embeddings_7k_full.zip or embeddings_7k_subset.zip
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple, Optional
import kagglehub

# Configuration constants
DATA_DIR = Path("data")
KAGGLE_DATASET = "dylanjcastillo/7k-books-with-metadata"

def _find_data_file(path: Path) -> Optional[Path]:
    """Find CSV or JSON file in the downloaded dataset directory."""
    for ext in ["*.csv", "*.json"]:
        files = sorted(path.rglob(ext))
        if files:
            return files[0]
    return None

def load_raw_data() -> pd.DataFrame:
    
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

def load_and_clean_data(use_cache: bool = True, cache_path: Optional[Path] = None) -> pd.DataFrame:
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

def select_high_quality_books(df: pd.DataFrame, target_count: int = 1200) -> pd.DataFrame:
    """
    Select a high-quality subset of books based on description quality scoring.
    """
    # Score all books
    df = df.copy()
    df['quality_score'] = df['description'].apply(score_description)

    # Sort by quality score (descending)
    df = df.sort_values('quality_score', ascending=False)

    # Take top quality books, but ensure we have good genre diversity
    top_books = df.head(target_count)

    # Reset index and remove quality score
    top_books = top_books.reset_index(drop=True)
    top_books = top_books.drop('quality_score', axis=1)

    return top_books

def clean_all_books_comprehensive(df: pd.DataFrame) -> pd.DataFrame:
  
    print(f"Starting comprehensive cleaning of {len(df)} books...")

    # data cleaning
    df = df.copy()

    # Fill missing essential fields
    df['title'] = df['title'].fillna('Unknown Title').astype(str).str.strip()
    df['author'] = df['author'].fillna('Unknown Author').astype(str).str.strip()
    df['description'] = df['description'].fillna('').astype(str).str.strip()

    # Remove books without meaningful content
    df = df[df['description'].str.len() >= 20]  
    df = df[df['title'].str.len() >= 3] 

    print(f"After basic filtering: {len(df)} books")

    print("Applying comprehensive genre and mood assignment...")
    df = assign_genres_and_moods(df)

    # Enhance descriptions to match quality of original 60 books
    print("Enhancing descriptions for semantic search quality...")
    df['description'] = df.apply(enhance_description, axis=1)

    df = df[df['genre'] != ''] 
    df = df[df['mood'] != '']  

    # Remove duplicates based on title+author
    df = df.drop_duplicates(subset=['title', 'author'], keep='first')

    df = df.reset_index(drop=True)
    print(f"Final cleaned dataset: {len(df)} books with complete metadata")

    return df

def score_description(text: str) -> float:
    """Score description quality based on length, vocabulary richness, and structure."""
    if not text or len(text.strip()) < 50:
        return 0

    score = 0

    # Length bonus (up to 500 chars)
    length = min(len(text), 500)
    score += length / 10  # Max 50 points

    # Vocabulary richness (unique words)
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = len(set(words))
    score += min(unique_words / 5, 20) 

    # Structure bonus
    sentences = len(re.split(r'[.!?]+', text))
    score += min(sentences, 10) 

    # Content indicators
    content_keywords = ['plot', 'story', 'character', 'setting', 'theme', 'conflict']
    keyword_count = sum(1 for keyword in content_keywords if keyword in text.lower())
    score += keyword_count * 2  

    return score

def assign_genres_and_moods(df: pd.DataFrame) -> pd.DataFrame:

    genre_keywords = {
        'Fantasy': ['magic', 'wizard', 'dragon', 'kingdom', 'quest', 'sword', 'elves'],
        'Sci-Fi': ['space', 'alien', 'robot', 'future', 'technology', 'scientist'],
        'Romance': ['love', 'relationship', 'marriage', 'heart', 'passion'],
        'Mystery': ['detective', 'murder', 'crime', 'investigation', 'clue'],
        'Horror': ['ghost', 'haunted', 'monster', 'terror', 'fear', 'supernatural'],
        'Literary': ['society', 'identity', 'philosophy', 'meaning', 'life'],
        'Historical': ['war', 'history', 'period', 'century', 'king', 'queen']
    }

    mood_keywords = {
        'Adventurous': ['adventure', 'quest', 'journey', 'explore', 'hero', 'brave'],
        'Dark': ['dark', 'murder', 'death', 'evil', 'horror', 'tragic'],
        'Emotional': ['love', 'heart', 'feeling', 'passion', 'sad', 'joy'],
        'Suspenseful': ['suspense', 'mystery', 'tension', 'thriller'],
        'Inspiring': ['inspire', 'hope', 'dream', 'achieve', 'overcome'],
        'Reflective': ['think', 'philosophy', 'meaning', 'identity', 'soul']
    }

    def assign_genre(desc: str) -> str:
        desc_lower = desc.lower()
        for genre, keywords in genre_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return genre
        return 'Fiction'

    def assign_mood(desc: str) -> str:
        desc_lower = desc.lower()
        for mood, keywords in mood_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return mood
        return 'General'

    df = df.copy()
    df['genre'] = df['description'].apply(assign_genre)
    df['mood'] = df['description'].apply(assign_mood)

    return df

def enhance_description(row) -> str:
    """
    Enhance book descriptions to match the quality and detail level of the original 60 books.
    Creates richer, more semantic-search friendly descriptions.
    """
    title = row['title']
    author = row['author']
    genre = row['genre']
    mood = row['mood']
    desc = row['description']

   
    if len(desc) > 150 and any(keyword in desc.lower() for keyword in
                               ['story', 'novel', 'tale', 'journey', 'world', 'character']):
        return desc

    # Build enhanced description based on available metadata
    enhanced_parts = []

    # Add genre/mood context
    genre_context = {
        'Fantasy': f'In a fantastical {genre.lower()} world filled with magic and wonder,',
        'Sci-Fi': f'In a futuristic {genre.lower()} setting exploring science and technology,',
        'Mystery': f'In a suspenseful {genre.lower()} filled with intrigue and secrets,',
        'Romance': f'In a heartfelt {genre.lower()} exploring love and relationships,',
        'Horror': f'In a chilling {genre.lower()} filled with terror and darkness,',
        'Literary': f'In a thought-provoking {genre.lower()} examining the human condition,',
        'Historical': f'In a meticulously researched {genre.lower()} set in historical times,',
        'Contemporary': f'In a modern {genre.lower()} reflecting contemporary life,',
        'Memoir': f'In an intimate {genre.lower()} sharing personal experiences and growth,',
        'Young Adult': f'In a coming-of-age {genre.lower()} exploring youth and self-discovery,'
    }

    mood_context = {
        'Adventurous': 'an epic adventure unfolds',
        'Dark': 'shadowy themes emerge',
        'Emotional': 'deep emotions are explored',
        'Hopeful': 'themes of hope and resilience shine through',
        'Inspiring': 'inspirational messages resonate',
        'Suspenseful': 'tension builds throughout',
        'Intellectual': 'intellectual depth challenges readers',
        'Whimsical': 'whimsical elements bring joy and wonder',
        'Reflective': 'moments of reflection provoke thought'
    }

    # Start with genre context
    if genre in genre_context:
        enhanced_parts.append(genre_context[genre])

    # Add mood context
    if mood in mood_context:
        enhanced_parts.append(mood_context[mood])

    # Add the original description
    if desc:
        enhanced_parts.append(desc.strip())
    else:
        # Fallback description based on genre/mood
        fallback_desc = f"This {genre.lower()} {mood.lower()} story follows compelling characters through engaging experiences that explore the depths of human experience and imagination."
        enhanced_parts.append(fallback_desc)

    # Add author context if available
    if author != 'Unknown Author':
        enhanced_parts.append(f"Written by {author}, this work showcases their unique voice and perspective.")

    enhanced_desc = " ".join(enhanced_parts)

    # Ensure minimum length and quality
    if len(enhanced_desc) < 100:
        enhanced_desc += f" The narrative weaves themes of {mood.lower()} exploration within the {genre.lower()} genre, creating an engaging reading experience."

    return enhanced_desc

def generate_sentence_transformers_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: Sentence Transformers not installed. Install with: pip install sentence-transformers")
        raise

    print(f"GENERATING: Generating Sentence Transformers embeddings for {len(texts)} books...")
    print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Batch size: {batch_size}")

    # Load the model (this will download ~90MB on first run)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    all_embeddings = []

    # Process in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")

        # Generate embeddings for this batch
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

        print(f"     SUCCESS: Batch {batch_num} completed")

    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    print(f"SUCCESS: Generated Sentence Transformers embeddings: {embeddings_array.shape}")
    print(f"   Dimensions: {embeddings_array.shape[1]} (sentence-transformers/all-MiniLM-L6-v2)")
    print(f"   Books processed: {embeddings_array.shape[0]}")

    return embeddings_array

def main(use_all_books: bool = True, embedding_method: str = "auto"):
    """Main workflow to create embeddings from 7k dataset.

    Args:
        use_all_books: If True, clean and process ALL books from the dataset.
                      If False, select top quality subset only.
        embedding_method: "auto" (Sentence Transformers),
                         "sentence-transformers" (force Sentence Transformers)
    """
    # Determine embedding method
    if embedding_method == "auto":
        print("AUTO-SELECT: Auto-selecting embedding method...")
        print("   AUTO-SELECT: Defaulting to Sentence Transformers (local, no API required)")
        actual_method = "sentence-transformers"
    else:
        actual_method = embedding_method

    if use_all_books:
        print("PROCESSING: Processing ALL Books from 7K Dataset (Comprehensive Clean)")
        print(f"INFO: Using {actual_method.upper()} embeddings")
    else:
        print("PROCESSING: Creating High-Quality Book Subset from 7K Dataset")
        print(f"INFO: Using {actual_method.upper()} embeddings")

    print("=" * 70)

    try:
        # Load the full 7k dataset
        print("DOWNLOADING: Loading 7k books dataset from Kaggle...")
        df = load_and_clean_data(use_cache=False)  # Force fresh download
        print(f"SUCCESS: Loaded {len(df)} books from Kaggle dataset")

        # Comprehensive cleaning of ALL books
        print("CLEANING: Applying comprehensive cleaning to ALL books...")
        if use_all_books:
            cleaned_df = clean_all_books_comprehensive(df)
            final_df = cleaned_df  # Use all cleaned books
        else:
            # First apply comprehensive cleaning, then select top quality subset
            cleaned_df = clean_all_books_comprehensive(df)
            final_df = select_high_quality_books(cleaned_df, target_count=1200)

        # Genres and moods are already assigned in comprehensive cleaning

        # Show statistics
        print(f"\nSTATS: Dataset Statistics:")
        print(f"   Total books: {len(final_df)}")
        print(f"   Genres: {final_df['genre'].value_counts().head(5).to_dict()}")
        print(f"   Moods: {final_df['mood'].value_counts().head(5).to_dict()}")

        # Show sample
        print("\nSAMPLES: Sample books:")
        for _, row in final_df.head(3).iterrows():
            print(f"  • '{row['title']}' by {row['author']}")
            print(f"    Genre: {row['genre']} | Mood: {row['mood']}")
            print(f"    Description: {row['description'][:120]}...")

        # Generate embeddings based on selected method
        descriptions = final_df['description'].tolist()

        if actual_method == "sentence-transformers":
            print(f"\nGENERATING: Generating Sentence Transformers embeddings for {len(final_df)} books...")
            embeddings_array = generate_sentence_transformers_embeddings(descriptions, batch_size=32)
        else:
            raise ValueError(f"Unknown embedding method: {actual_method}")

        embeddings = embeddings_array.tolist()  # Convert to list format for JSON serialization

        # Create embeddings package
        output_dir = Path("precomputed_embeddings")
        output_dir.mkdir(exist_ok=True)

        # Save books metadata
        books_data = []
        for _, row in final_df.iterrows():
            book = {
                "title": str(row['title']),
                "author": str(row.get('author', 'Unknown')),
                "genre": str(row['genre']),
                "mood": str(row['mood']),
                "description": str(row['description'])
            }
            books_data.append(book)

        with open(output_dir / "books_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(books_data, f, indent=2, ensure_ascii=False)

        # Save embeddings
        embeddings_array = np.array(embeddings)
        np.save(output_dir / "embeddings.npy", embeddings_array)

        print("\nSUCCESS: Created embeddings package:")
        print(f"   BOOKS: Books: {len(books_data)}")
        print(f"   SHAPE: Embeddings shape: {embeddings_array.shape}")
        print(f"   SAVED: Saved to: {output_dir}")

        # Create zip for easy upload
        import zipfile
        zip_name = "embeddings_7k_full.zip" if use_all_books else "embeddings_7k_subset.zip"
        zip_path = Path(zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    zip_file.write(file_path, file_path.relative_to(output_dir.parent))

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"PACKAGE: Created upload package: {zip_path} ({zip_size_mb:.1f} MB)")

        dataset_type = "full_7k" if use_all_books else "subset_1200"
        print(f"\nNEXT STEPS: Next Steps for {dataset_type} dataset:")
        print("   1. Upload embeddings_7k_full.zip to Google Drive")
        print("   2. Make it publicly accessible")
        print("   3. Copy the download link and update embeddings/store.py")
        print("   4. Test with: python main.py")

    except Exception as e:
        print(f"ERROR: Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
  
    import argparse
    parser = argparse.ArgumentParser(description="Generate book embeddings from 7k dataset")
    parser.add_argument('dataset', choices=['all', 'subset'], default='all',
                       help='Process all books or quality subset')
    parser.add_argument('embeddings', choices=['auto', 'sentence-transformers'], default='sentence-transformers',
                       help='Embedding method: auto (Sentence Transformers), sentence-transformers (recommended)')

    args = parser.parse_args()

    use_all_books = (args.dataset == 'all')
    embedding_method = args.embeddings

    print("Command line options:")
    print(f"  Dataset: {'ALL books' if use_all_books else 'Quality subset'}")
    print(f"  Embeddings: {embedding_method.upper()}")
    print()

    main(use_all_books=use_all_books, embedding_method=embedding_method)