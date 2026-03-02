"""
Embedding & Storage: Load precomputed embeddings and store in ChromaDB.
Supports Sentence Transformers, OpenAI, and TF-IDF embeddings.
No API keys required - fully offline operation after initial download.
"""

import hashlib
import json
import time
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    PROJECT_ROOT,
)

# Configuration for precomputed embeddings
# Option 1: Download from Google Drive (default)
EMBEDDINGS_URL = "https://drive.google.com/uc?id=1TYJ-GBdc7_e058-s0EZGMeMbRkCwQqe6"  # Sentence Transformers embeddings

# Option 2: Use local file (fallback)
# EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings_7k_full"  # Point to extracted local folder

EMBEDDINGS_DIR = PROJECT_ROOT / "precomputed_embeddings"
BOOKS_DATA_FILE = EMBEDDINGS_DIR / "books_metadata.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"


def download_precomputed_embeddings(url: str = EMBEDDINGS_URL) -> Path:
    """
    Download precomputed embeddings from Google Drive with robust error handling.
    Includes retry logic, progress indicators, and timeout handling for large files.
    Falls back to local ZIP files if available.

    Args:
        url: Google Drive public download URL

    Returns:
        Path to the embeddings directory
    """
    # Check for local zip file first (transferred via USB/cloud)
    local_zip = PROJECT_ROOT / "embeddings_7k_full.zip"
    if local_zip.exists():
        print(f"📦 Using local embeddings file: {local_zip}")

        # Create embeddings directory
        EMBEDDINGS_DIR.mkdir(exist_ok=True)

        try:
            # Extract local zip file
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(EMBEDDINGS_DIR)

            print(f"✅ Local embeddings extracted to {EMBEDDINGS_DIR}")
            return EMBEDDINGS_DIR

        except Exception as e:
            print(f"❌ Local extraction failed: {e}")

    # Fall back to Google Drive download
    if not url:
        raise RuntimeError("No URL provided and no local embeddings_7k_full.zip found")

    print("📥 Downloading precomputed embeddings from Google Drive...")

    # Create embeddings directory
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # Try multiple times with increasing timeouts
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"   Attempt {attempt + 1}/{max_retries}...")

            # Longer timeout for large files
            timeout = 300  # 5 minutes
            headers = {'User-Agent': 'Mozilla/5.0'}

            # Download the zip file with progress
            response = requests.get(url, stream=True, timeout=timeout, headers=headers)
            response.raise_for_status()

            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            print(f"   Downloading {total_size / (1024*1024):.1f} MB...")

            # Download and save to memory
            content = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content.write(chunk)
                    downloaded_size += len(chunk)

                    # Simple progress indicator
                    if downloaded_size % (50*1024*1024) == 0 and total_size > 0:  # Every 50MB
                        progress = (downloaded_size / total_size) * 100
                        print(f"   Downloaded: {progress:.0f}% ({downloaded_size/(1024*1024):.0f}MB)", flush=True)

            print("   Download complete. Extracting...")

            # Extract zip file
            content.seek(0)
            with zipfile.ZipFile(content) as zip_ref:
                zip_ref.extractall(EMBEDDINGS_DIR)

            print(f"\n✅ Embeddings downloaded and extracted to {EMBEDDINGS_DIR}")
            return EMBEDDINGS_DIR

        except requests.exceptions.Timeout:
            wait_time = 10 * (attempt + 1)
            print(f"   ⏳ Timeout (attempt {attempt + 1}). Waiting {wait_time}s...")
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:
            wait_time = 5 * (attempt + 1)
            print(f"   ⚠️ Connection error (attempt {attempt + 1}): {e}")
            print(f"   ⏳ Waiting {wait_time}s...")
            time.sleep(wait_time)

        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download embeddings after {max_retries} attempts: {e}")
            else:
                wait_time = 3 * (attempt + 1)
                print(f"   ⚠️ Error (attempt {attempt + 1}): {e}")
                print(f"   ⏳ Waiting {wait_time}s...")
                time.sleep(wait_time)


def load_books_metadata() -> List[Dict[str, Any]]:
    """
    Load book metadata from the downloaded embeddings.
    Returns list of book dictionaries with title, author, genre, description, etc.
    """
    if not BOOKS_DATA_FILE.exists():
        raise FileNotFoundError(f"Books metadata file not found: {BOOKS_DATA_FILE}")

    with open(BOOKS_DATA_FILE, 'r', encoding='utf-8') as f:
        books_data = json.load(f)

    print(f"📚 Loaded metadata for {len(books_data)} books")
    return books_data


def load_embeddings() -> np.ndarray:
    """
    Load precomputed embeddings from numpy file.
    Supports both OpenAI embeddings (1536-dim) and TF-IDF embeddings (variable-dim).
    """
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"🔢 Loaded embeddings with shape {embeddings.shape}")

    # Validate embedding dimensions
    if embeddings.shape[1] == 1536:
        print("   Detected: OpenAI text-embedding-3-small embeddings")
    elif embeddings.shape[1] == 384:
        print("   Detected: Sentence Transformers all-MiniLM-L6-v2 embeddings")
    elif embeddings.shape[1] <= 2000:
        print(f"   Detected: TF-IDF embeddings ({embeddings.shape[1]} features)")
    else:
        print(f"   Detected: Custom embeddings ({embeddings.shape[1]} dimensions)")

    return embeddings


class EmbeddingStore:
    """
    Manages loading of precomputed embeddings and ChromaDB storage.
    Downloads embeddings from Google Drive and loads them into ChromaDB.
    No API keys required - fully reproducible.
    """

    def __init__(self):
        self.chroma_dir = CHROMA_DIR
        self.collection_name = COLLECTION_NAME
        self._client = None
        self._collection = None
        self._books_data = None
        self._embeddings = None

    def _get_client(self):
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    def _ensure_embeddings_downloaded(self):
        """Download precomputed embeddings if not already present, or use local files."""
        # Priority 1: Check for local extracted files (fastest)
        local_embeddings_dir = PROJECT_ROOT / "embeddings_7k_full"
        local_books_file = local_embeddings_dir / "books_metadata.json"
        local_embeddings_file = local_embeddings_dir / "embeddings.npy"

        if (local_embeddings_dir.exists() and
            local_books_file.exists() and
            local_embeddings_file.exists()):
            print("✅ Using local embeddings (fastest option)")
            # Update paths to use local files
            global BOOKS_DATA_FILE, EMBEDDINGS_FILE
            BOOKS_DATA_FILE = local_books_file
            EMBEDDINGS_FILE = local_embeddings_file
            return

        # Priority 2: Check for cached downloaded embeddings
        if (EMBEDDINGS_DIR.exists() and
            BOOKS_DATA_FILE.exists() and
            EMBEDDINGS_FILE.exists()):
            print("✅ Using cached downloaded embeddings")
            return

        # Priority 3: Try to download from Google Drive
        print("🔄 No local embeddings found. Attempting download...")
        try:
            download_precomputed_embeddings()
            print("✅ Successfully downloaded embeddings from Google Drive")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download embeddings from Google Drive. "
                f"Please download embeddings_7k_full.zip manually and extract to 'embeddings_7k_full' folder. "
                f"Google Drive link: https://drive.google.com/uc?id=1TYJ-GBdc7_e058-s0EZGMeMbRkCwQqe6 "
                f"Error: {e}"
            )

    def load_precomputed_store(self) -> 'EmbeddingStore':
        """
        Load precomputed embeddings from Google Drive and create ChromaDB collection.
        This is the main entry point for the reproducible version.
        """
        print("🚀 Loading Semantic Book Recommender with precomputed embeddings...")

        # Step 1: Download embeddings if needed
        self._ensure_embeddings_downloaded()

        # Step 2: Load books metadata
        self._books_data = load_books_metadata()

        # Step 3: Load embeddings
        self._embeddings = load_embeddings()

        # Step 4: Create ChromaDB collection with precomputed embeddings
        self._create_chroma_collection_from_precomputed()

        print("🎉 Successfully loaded precomputed embeddings into ChromaDB!")
        return self

    def _create_chroma_collection_from_precomputed(self):
        """Create ChromaDB collection using precomputed embeddings."""
        client = self._get_client()

        # Delete existing collection if it exists
        try:
            client.delete_collection(self.collection_name)
            print(f"🗑️ Cleared existing collection: {self.collection_name}")
        except ValueError:
            pass  # Collection didn't exist

        # Create new collection for VECTOR SEARCH
        # ChromaDB will use your PRECOMPUTED embeddings for similarity calculations
        collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for embeddings
        )

        # Prepare data for ChromaDB using precomputed embeddings
        ids = [f"book_{i}" for i in range(len(self._books_data))]
        documents = [book["description"] for book in self._books_data]  # Use description as the main text
        metadatas = []

        for book in self._books_data:
            # Create metadata for each book (excluding description to avoid duplication)
            metadata = {
                "title": book["title"],
                "author": book.get("author", "Unknown"),
                "genre": book["genre"],
                "mood": book.get("mood", "General"),
                "description": book["description"]  # Include for LangChain access
            }
            metadatas.append(metadata)

        # Convert embeddings to list format for ChromaDB
        embeddings_list = self._embeddings.tolist()

        # Add all data to ChromaDB with PRECOMPUTED embeddings
        print(f"📥 Adding {len(documents)} books to ChromaDB collection...")
        collection.add(
            ids=ids,                           # Unique IDs for each book
            embeddings=embeddings_list,        # Your precomputed OpenAI vectors
            documents=documents,               # Original book descriptions
            metadatas=metadatas                # Title, author, genre, mood, etc.
        )

        print(f"✅ Created ChromaDB collection: {self.collection_name} with {len(documents)} books")
        self._collection = collection

    def get_vectorstore(self):
        """Get LangChain-compatible Chroma vectorstore for querying."""
        if not self._get_collection():
            raise RuntimeError("ChromaDB collection not initialized. Call load_precomputed_store() first.")

        # Import here to avoid circular imports
        from langchain_community.vectorstores import Chroma as LangChainChroma

        # Create LangChain wrapper - we don't need an embedding function since we're using precomputed embeddings
        return LangChainChroma(
            client=self._get_client(),
            collection_name=self.collection_name,
            embedding_function=None  # Precomputed embeddings don't need a live embedding function
        )

    def get_books_data(self) -> List[Dict[str, Any]]:
        """Get the loaded books metadata."""
        return self._books_data or []


def get_or_create_store(
    df: pd.DataFrame = None,
    force_rebuild: bool = False,
) -> EmbeddingStore:
    """
    Convenience function to get a store with precomputed embeddings.
    For the reproducible version, this ignores the df parameter and loads precomputed data.
    """
    store = EmbeddingStore()
    store.load_precomputed_store()
    return store