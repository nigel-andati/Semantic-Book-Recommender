#!/usr/bin/env python3
"""
Local Development Version of the Semantic Book Recommender.
Mirrors the Colab notebook structure for local testing and development.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import gradio as gr

def main():
    """Run the complete book recommender locally."""

    print("🚀 Starting Semantic Book Recommender (Local Development)")
    print("=" * 60)

    # Check for embeddings
    embeddings_dir = Path("embeddings_7k_full")
    if not (embeddings_dir.exists() and
            (embeddings_dir / "books_metadata.json").exists() and
            (embeddings_dir / "embeddings.npy").exists()):
        print("❌ Embeddings not found!")
        print("   Run: python create_embeddings_subset.py all sentence-transformers")
        return

    print("✅ Found local embeddings")

    # Load data (matches notebook step 3)
    print("📚 Loading book metadata...")
    with open("embeddings_7k_full/books_metadata.json", 'r', encoding='utf-8') as f:
        books_data = json.load(f)
    print(f"✅ Loaded {len(books_data)} books")

    print("🔢 Loading embeddings...")
    embeddings_array = np.load("embeddings_7k_full/embeddings.npy")
    print(f"✅ Loaded embeddings: {embeddings_array.shape}")

    print("🤖 Loading Sentence Transformers model...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Model loaded")

    # Setup ChromaDB (matches notebook step 3)
    print("🗄️ Setting up ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = chroma_client.get_collection(name="book_recommendations")
        print("✅ Using existing ChromaDB collection")
    except:
        print("📝 Creating new ChromaDB collection...")
        collection = chroma_client.create_collection(name="book_recommendations")

        for i, book in enumerate(books_data):
            collection.add(
                ids=[f"book_{i}"],
                embeddings=[embeddings_array[i].tolist()],
                metadatas=[book]
            )
            if (i + 1) % 1000 == 0:
                print(f"   Added {i + 1}/{len(books_data)} books")

        print("✅ Collection created")

    # Analysis setup (matches notebook step 4)
    print("🧠 Setting up analysis system...")
    try:
        classifier = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            device=-1)
        print("✅ Zero-shot classifier ready")
    except Exception as e:
        print(f"⚠️ Classifier setup failed: {e}")
        classifier = None

    sentiment_analyzer = None  # Disabled for performance

    # Emotion/Theme/Genre labels (matches notebook)
    EMOTION_LABELS = ["hopeful", "sad", "dark", "comforting", "uplifting", "mysterious",
                     "suspenseful", "romantic", "inspiring", "melancholic", "emotional",
                     "reflective", "introspective", "optimistic", "haunting"]

    THEME_LABELS = ["identity", "justice", "trauma", "colonialism", "migration",
                   "self-discovery", "healing", "belonging", "resistance",
                   "cultural conflict", "moral dilemmas", "existential themes",
                   "personal transformation", "human nature"]

    GENRE_LABELS = ["fiction", "non-fiction", "romance", "mystery", "thriller",
                   "science fiction", "fantasy", "historical fiction", "biography",
                   "memoir", "literary fiction", "horror", "young adult",
                   "self-help", "philosophy", "political", "crime", "adventure",
                   "drama", "comedy", "contemporary"]

    ATMOSPHERE_LABELS = ["mysterious", "suspenseful", "eerie", "tense", "foreboding",
                        "adventurous", "epic journey", "quest-driven", "introspective",
                        "character-driven", "emotionally immersive", "fast-paced thriller",
                        "slow reflective narrative", "lyrical", "dreamy", "gritty",
                        "surreal", "claustrophobic", "intimate", "meditative"]

    def analyze_text(text: str):
        """Analyze text for classification (matches notebook)."""
        if not text or len(text.strip()) < 10:
            return {"emotions": [], "themes": [], "genres": [], "atmosphere": []}

        result = {}
        if classifier:
            try:
                # Emotions
                emotion_result = classifier(text[:512], EMOTION_LABELS, multi_label=True)
                result["emotions"] = [
                    {"label": label, "score": score}
                    for label, score in zip(emotion_result["labels"][:3], emotion_result["scores"][:3])
                    if score > 0.3
                ]

                # Themes
                theme_result = classifier(text[:512], THEME_LABELS, multi_label=True)
                result["themes"] = [
                    {"label": label, "score": score}
                    for label, score in zip(theme_result["labels"][:2], theme_result["scores"][:2])
                    if score > 0.4
                ]

                # Genres
                genre_result = classifier(text[:512], GENRE_LABELS, multi_label=True)
                result["genres"] = [
                    {"label": label, "score": score}
                    for label, score in zip(genre_result["labels"][:2], genre_result["scores"][:2])
                    if score > 0.5
                ]

                # Atmosphere
                atmosphere_result = classifier(text[:512], ATMOSPHERE_LABELS, multi_label=True)
                result["atmosphere"] = [
                    {"label": label, "score": score}
                    for label, score in zip(atmosphere_result["labels"][:2], atmosphere_result["scores"][:2])
                    if score > 0.3
                ]

            except Exception as e:
                print(f"Analysis error: {e}")
                result = {"emotions": [], "themes": [], "genres": [], "atmosphere": []}

        return result

    # Search function (matches notebook step 5)
    def search_books(query, emotion_filter="None", theme_filter="None",
                    genre_filter="None", atmosphere_filter="None",
                    mood_filter="None", sentiment_filter="None", max_results=8):

        if not query.strip() and all(f == "None" for f in [emotion_filter, theme_filter, genre_filter, atmosphere_filter, mood_filter, sentiment_filter]):
            return "Please enter a query or select at least one filter."

        try:
            # Build enhanced query
            query_parts = [query.strip()]
            if emotion_filter != "None": query_parts.append(f"{emotion_filter} emotion")
            if theme_filter != "None": query_parts.append(f"{theme_filter} theme")
            if genre_filter != "None": query_parts.append(f"{genre_filter} genre")
            if atmosphere_filter != "None": query_parts.append(f"{atmosphere_filter} atmosphere")
            if mood_filter != "None": query_parts.append(f"{mood_filter} mood")
            if sentiment_filter != "None": query_parts.append(f"{sentiment_filter} sentiment")

            enhanced_query = " ".join(query_parts)
            print(f"🔍 Searching for: '{enhanced_query}'")

            # Generate embedding and search
            if query.strip():
                query_embedding = embed_model.encode(query).tolist()
                results = collection.query(query_embeddings=[query_embedding], n_results=50)
                candidate_ids = results["ids"][0] if results["ids"] else []
            else:
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
            active_filters = []
            if emotion_filter != "None": active_filters.append(f"emotion: {emotion_filter}")
            if theme_filter != "None": active_filters.append(f"theme: {theme_filter}")
            if genre_filter != "None": active_filters.append(f"genre: {genre_filter}")
            if atmosphere_filter != "None": active_filters.append(f"atmosphere: {atmosphere_filter}")
            if mood_filter != "None": active_filters.append(f"mood: {mood_filter}")
            if sentiment_filter != "None": active_filters.append(f"sentiment: {sentiment_filter}")

            if active_filters:
                summary += f" (filtered by: {', '.join(active_filters)})"

            return f"{summary}\n\n" + "\n\n".join(processed_results)

        except Exception as e:
            return f"Search error: {str(e)}"

    print("🔍 Search function ready!")

    # Test search
    print("🧪 Testing search function...")
    test_result = search_books("fantasy adventure", max_results=2)
    print("✅ Search test completed")

    # Create Gradio interface (matches notebook step 6)
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
                    "None", "Fiction", "Romance", "Mystery", "Thriller", "Fantasy",
                    "Sci-Fi", "Historical", "Biography", "Horror"
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

if __name__ == "__main__":
    main()