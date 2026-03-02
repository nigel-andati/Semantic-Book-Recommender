#!/usr/bin/env python3
"""
Main application for the Semantic Book Recommender.
Runs the Gradio interface locally for development and testing.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

def check_embeddings():
    """Check if embeddings are available locally."""
    embeddings_dir = Path("embeddings_7k_full")
    books_file = embeddings_dir / "books_metadata.json"
    embeddings_file = embeddings_dir / "embeddings.npy"

    return (embeddings_dir.exists() and
            books_file.exists() and
            embeddings_file.exists())

def main():
    """Main application entry point."""
    print("🚀 Starting Semantic Book Recommender (Local Development)")
    print("=" * 60)

    # Check if embeddings exist
    if not check_embeddings():
        print("❌ Embeddings not found locally.")
        print("   Run: python create_embeddings_subset.py all sentence-transformers")
        print("   Or download from Google Drive and extract to embeddings_7k_full/")
        return

    print("✅ Local embeddings found")

    try:
        # Import after checking dependencies
        import json
        import numpy as np
        import chromadb
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline
        import gradio as gr

        # Load data (similar to notebook)
        print("📚 Loading book metadata...")
        with open("embeddings_7k_full/books_metadata.json", 'r', encoding='utf-8') as f:
            books_data = json.load(f)
        print(f"✅ Loaded {len(books_data)} books")

        print("🔢 Loading embeddings...")
        embeddings_array = np.load("embeddings_7k_full/embeddings.npy")
        print(f"✅ Loaded embeddings: {embeddings_array.shape}")

        # Initialize Sentence Transformers
        print("🤖 Loading Sentence Transformers model...")
        embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded")

        # Setup ChromaDB
        print("🗄️ Setting up ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Create/load collection
        try:
            collection = chroma_client.get_collection(name="book_recommendations")
            print("✅ Using existing ChromaDB collection")
        except:
            collection = chroma_client.create_collection(name="book_recommendations")
            print("📝 Setting up new collection...")

            # Add books to collection
            for i, book in enumerate(books_data):
                collection.add(
                    ids=[f"book_{i}"],
                    embeddings=[embeddings_array[i].tolist()],
                    metadatas=[book]
                )

                if (i + 1) % 500 == 0:
                    print(f"   Added {i + 1}/{len(books_data)} books")

            print("✅ Collection created")

        # Initialize analysis components
        print("🧠 Setting up analysis...")
        try:
            classifier = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=-1)
            print("✅ Classifier ready")
        except:
            classifier = None
            print("⚠️ Classifier not available")

        sentiment_analyzer = None  # Disabled for performance

        def analyze_text(text: str) -> dict:
            """Analyze text for emotions, themes, genres."""
            if not text or len(text.strip()) < 10:
                return {"emotions": [], "themes": [], "genres": [], "atmosphere": [], "sentiment": {"neutral": 0.5}}

            result = {}
            if classifier:
                try:
                    emotion_result = classifier(text[:512], [
                        "hopeful", "sad", "dark", "comforting", "uplifting", "mysterious",
                        "suspenseful", "romantic", "inspiring", "melancholic", "emotional",
                        "reflective", "introspective", "optimistic", "haunting"
                    ], multi_label=True)
                    result["emotions"] = [
                        {"label": label, "score": score}
                        for label, score in zip(emotion_result["labels"][:3], emotion_result["scores"][:3])
                        if score > 0.3
                    ]
                except:
                    result["emotions"] = []

            return result

        def search_books(query, emotion_filter="None", theme_filter="None",
                        genre_filter="None", atmosphere_filter="None",
                        mood_filter="None", sentiment_filter="None", max_results=8):

            if not query.strip() and all(f == "None" for f in [emotion_filter, theme_filter, genre_filter, atmosphere_filter, mood_filter, sentiment_filter]):
                return "Please enter a query or select at least one filter."

            try:
                # Build query
                query_parts = [query.strip()]
                if emotion_filter != "None": query_parts.append(f"{emotion_filter} emotion")
                if theme_filter != "None": query_parts.append(f"{theme_filter} theme")
                if genre_filter != "None": query_parts.append(f"{genre_filter} genre")
                if atmosphere_filter != "None": query_parts.append(f"{atmosphere_filter} atmosphere")
                if mood_filter != "None": query_parts.append(f"{mood_filter} mood")
                if sentiment_filter != "None": query_parts.append(f"{sentiment_filter} sentiment")

                enhanced_query = " ".join(query_parts)
                print(f"🔍 Searching for: '{enhanced_query}'")

                # Generate embedding
                if query.strip():
                    query_embedding = embed_model.encode(query).tolist()
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=max_results
                    )
                    candidate_ids = results["ids"][0] if results["ids"] else []
                else:
                    # Get all books if no query
                    all_books = collection.get()
                    candidate_ids = all_books["ids"] or []
                    results = {
                        "ids": [candidate_ids],
                        "distances": [[0] * len(candidate_ids)],
                        "metadatas": [all_books.get("metadatas", [{}]*len(candidate_ids))]
                    }

                if not candidate_ids:
                    return "No books found matching your criteria."

                # Process results
                processed_results = []
                for i, (doc_id, distance, metadata) in enumerate(zip(
                    results["ids"][0][:max_results],
                    results["distances"][0][:max_results],
                    results["metadatas"][0][:max_results]
                )):
                    relevance_score = max(0, 1 - (distance / 2))
                    relevance_pct = int(relevance_score * 100)

                    title = metadata.get("title", "Unknown Title")
                    author = metadata.get("author", "Unknown Author")
                    genre = metadata.get("genre", "Fiction")
                    mood = metadata.get("mood", "General")
                    description = metadata.get("description", "")[:150] + "..."

                    filled = relevance_pct // 10
                    relevance_bar = "█" * filled + "░" * (10 - filled)

                    result = f"📚 **{title}**\n👤 {author} | 🎭 {genre}\n📖 {description}\n🎯 {relevance_pct}% {relevance_bar}\n"

                    # Add analysis
                    if classifier:
                        analysis = analyze_text(description)
                        if analysis.get("emotions"):
                            emotions = [e["label"] for e in analysis["emotions"][:2]]
                            result += f"😊 Emotions: {', '.join(emotions)}\n"

                    processed_results.append(result)

                summary = f"Found {len(processed_results)} relevant books for '{query}'"
                return f"{summary}\n\n" + "\n\n".join(processed_results)

            except Exception as e:
                return f"Search error: {str(e)}"

        # Create Gradio interface
        print("🎨 Creating Gradio interface...")

        with gr.Blocks(title="Semantic Book Recommender", theme=gr.themes.Soft()) as demo:

            gr.Markdown("""
            # 📚 Semantic Book Recommender

            Discover your next read! Enter a description of what you're looking for,
            and our system will find books that match your interests.

            **Features:**
            - 7,000+ books in our database
            - Sentence Transformers for semantic understanding
            - Zero-shot classification (emotions, themes, genres)
            - Advanced filtering options
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="📝 What are you looking for?",
                        placeholder="e.g., 'mysterious adventure with dragons and magic'",
                        lines=3
                    )
                    max_results = gr.Slider(1, 20, value=8, step=1, label="📊 Number of results")

                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### Tips
                    - Be specific about themes, emotions, or genres
                    - Try: "uplifting story about overcoming adversity"
                    """)

            # Filters
            with gr.Tabs():
                with gr.TabItem("🎭 Genre"):
                    genre_filter = gr.Dropdown([
                        "None", "Fiction", "Romance", "Mystery", "Thriller",
                        "Fantasy", "Sci-Fi", "Historical", "Biography", "Horror"
                    ], value="None")

                with gr.TabItem("😊 Emotion"):
                    emotion_filter = gr.Dropdown([
                        "None", "Hopeful", "Sad", "Dark", "Comforting", "Uplifting",
                        "Mysterious", "Suspenseful", "Romantic", "Inspiring"
                    ], value="None")

                with gr.TabItem("🧠 Theme"):
                    theme_filter = gr.Dropdown([
                        "None", "Identity", "Justice", "Trauma", "Migration",
                        "Self-Discovery", "Healing", "Belonging"
                    ], value="None")

                with gr.TabItem("🌟 Atmosphere"):
                    atmosphere_filter = gr.Dropdown([
                        "None", "Mysterious", "Suspenseful", "Fast-paced", "Introspective"
                    ], value="None")

                with gr.TabItem("🎵 Mood"):
                    mood_filter = gr.Dropdown([
                        "None", "Adventurous", "Emotional", "Hopeful", "Reflective"
                    ], value="None")

            search_btn = gr.Button("🔍 Search Books", variant="primary", size="lg")
            results_output = gr.Textbox(
                label="📚 Recommended Books",
                lines=25,
                placeholder="Your recommendations will appear here..."
            )

            search_btn.click(
                fn=search_books,
                inputs=[query_input, emotion_filter, theme_filter, genre_filter,
                       atmosphere_filter, mood_filter, gr.State("None"), max_results],
                outputs=results_output
            )

            gr.Examples([
                ["fantasy adventure with magic and dragons"],
                ["heartwarming story about friendship"],
                ["dark mystery with suspense"],
                ["inspiring tale of overcoming adversity"]
            ], inputs=query_input)

        # Launch interface
        print("✅ Interface ready!")
        print("🌐 Opening local web interface...")
        demo.launch(share=False, show_error=True)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: python setup.py")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()