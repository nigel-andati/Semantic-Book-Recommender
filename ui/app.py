"""
Semantic Book Recommender - Gradio UI
Uses precomputed Sentence Transformers embeddings.
No API keys required - fully reproducible and local.
"""

import gradio as gr
from embeddings.store import get_or_create_store
from retrieval.retriever import get_retriever
import html


def recommend(query, use_genre, use_mood, genre, mood, use_emotion, emotion, use_theme, theme, use_atmosphere, atmosphere, use_sentiment, sentiment_preference):
    """
    Recommend books using precomputed embeddings and ChromaDB.

    Args:
        query: User's search query
        use_genre: Whether to filter by genre
        use_mood: Whether to filter by mood
        genre: Selected genre filter
        mood: Selected mood filter

    Returns:
        Formatted string with recommendations
    """
    if not query.strip():
        return "Please enter a description of the book you're looking for."

    try:
        # Get the precomputed store and retriever
        store = get_or_create_store()
        retriever = get_retriever(store.get_vectorstore(), k=10)

        # Load classification data for enhanced retrieval
        classification_data = None
        try:
            from data.pipeline import load_classified_books
            classified_df = load_classified_books(use_cache=True)
            # Convert to dict format for retriever
            classification_data = {}
            for _, row in classified_df.iterrows():
                book_id = f"{row['title']}_{row['author']}"  # Create unique ID
                classification_data[book_id] = {
                    "emotions": row.get("emotions", []),
                    "themes": row.get("themes", []),
                    "genres": row.get("genres", []),
                    "atmosphere": row.get("atmosphere", []),
                    "sentiment": row.get("sentiment", {})
                }
        except Exception as e:
            print(f"Classification data not available: {e}")

        # Build query with filters
        query_parts = [query.strip()]
        if use_genre:
            query_parts.append(f"{genre} genre")
        if use_mood:
            query_parts.append(f"{mood} mood")
        if use_emotion:
            query_parts.append(f"{emotion} emotion")
        if use_theme:
            query_parts.append(f"{theme} theme")
        if use_atmosphere:
            query_parts.append(f"{atmosphere} atmosphere")
        if use_sentiment:
            query_parts.append(f"{sentiment_preference} sentiment")

        combined_query = " ".join(query_parts)

        # Use enhanced retrieval if classification data is available
        if classification_data and hasattr(retriever, 'get_enhanced_recommendations'):
            enhanced_results = retriever.get_enhanced_recommendations(
                combined_query,
                classification_data=classification_data,
                semantic_weight=0.7,
                classification_weight=0.3
            )
            # Convert to expected format
            docs_with_scores = [(doc, score) for doc, score, _ in enhanced_results]
        else:
            # Fall back to basic similarity search
            docs_with_scores = retriever.get_relevant_documents_with_scores(combined_query)

        if not docs_with_scores:
            return f"No books found matching your criteria. Try:\n• Using fewer filters\n• Different keywords\n• More general descriptions"

        results = []
        for i, (doc, score) in enumerate(docs_with_scores[:10]):  # Show up to 10 results
            metadata = doc.metadata
            relevance_pct = int(score * 100)
            relevance_bar = "█" * (relevance_pct // 10) + "░" * (10 - relevance_pct // 10)

            result = f"📚 {metadata['title']}\n"
            result += f"🎭 Genre: {metadata['genre']} | Mood: {metadata['mood']}\n"
            result += f"📖 {metadata['description']}\n"
            result += f"🎯 Relevance: {relevance_pct}% {relevance_bar}\n"

            # Add classification info if available (from enhanced retrieval)
            if hasattr(doc, '_metadata') and 'classifications' in doc._metadata:
                classifications = doc._metadata['classifications']
                if classifications.get('emotions'):
                    emotions = [e.get('label', e) for e in classifications['emotions'][:2]]
                    result += f"😊 Emotions: {', '.join(emotions)}\n"
                if classifications.get('sentiment'):
                    sentiment = classifications['sentiment']
                    dominant = max(sentiment.items(), key=lambda x: x[1])
                    result += f"💭 Sentiment: {dominant[0].title()} ({dominant[1]:.1%})\n"

            results.append(result)

        summary = f"Found {len(docs_with_scores)} relevant books"
        filters_used = []
        if use_genre:
            filters_used.append(f"{genre}")
        if use_mood:
            filters_used.append(f"{mood} mood")
        if use_emotion:
            filters_used.append(f"{emotion} emotion")
        if use_theme:
            filters_used.append(f"{theme} theme")
        if use_atmosphere:
            filters_used.append(f"{atmosphere} atmosphere")
        if use_sentiment:
            filters_used.append(f"{sentiment_preference} sentiment")

        if filters_used:
            summary += f" (filtered by: {', '.join(filters_used)})"

        return f"{summary}\n\n" + "\n".join(results)

    except Exception as e:
        return f"Error: {str(e)}. Please ensure precomputed embeddings are properly downloaded."


def create_app():
    """
    Create and return the Gradio app.
    Loads precomputed embeddings automatically.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# 📚 Semantic Book Recommender")
        gr.Markdown("Find books using AI-powered semantic search. No API keys required!")

        query = gr.Textbox(
            placeholder="Describe the kind of book you want (e.g., 'magical adventure with wizards and dragons')...",
            lines=3,
            label="📝 What are you looking for?"
        )

        with gr.Tabs():
            with gr.Tab("🎭 Genre Filter"):
                use_genre = gr.Checkbox(label="Apply genre filter", value=False)
                genre = gr.Dropdown(
                    ["Fantasy", "Sci-Fi", "Dystopian", "Romance", "Mystery", "Thriller", "Philosophical", "Literary", "Horror", "Contemporary", "Historical", "Memoir", "Young Adult"],
                    value="Fantasy",
                    label="Select Genre"
                )

            with gr.Tab("🎵 Mood Filter"):
                use_mood = gr.Checkbox(label="Apply mood filter", value=False)
                mood = gr.Dropdown(
                    ["Adventurous", "Dark", "Emotional", "Hopeful", "Whimsical", "Reflective", "Suspenseful", "Intellectual", "Inspiring"],
                    value="Adventurous",
                    label="Select Mood"
                )

            with gr.Tab("😊 Emotion Filter"):
                use_emotion = gr.Checkbox(label="Apply emotion filter", value=False)
                emotion = gr.Dropdown(
                    ["Hopeful", "Sad", "Dark", "Comforting", "Uplifting", "Mysterious", "Suspenseful", "Romantic", "Inspiring", "Melancholic"],
                    value="Hopeful",
                    label="Select Emotional Tone"
                )

            with gr.Tab("🧠 Theme Filter"):
                use_theme = gr.Checkbox(label="Apply theme filter", value=False)
                theme = gr.Dropdown(
                    ["Identity", "Justice", "Trauma", "Colonialism", "Migration", "Self-Discovery", "Healing", "Belonging", "Resistance", "Cultural Conflict"],
                    value="Identity",
                    label="Select Thematic Focus"
                )

            with gr.Tab("🌟 Atmosphere Filter"):
                use_atmosphere = gr.Checkbox(label="Apply atmosphere filter", value=False)
                atmosphere = gr.Dropdown(
                    ["Mysterious", "Suspenseful", "Fast-paced", "Slow & Reflective", "Introspective", "Character-driven", "Epic Journey", "Quest-driven"],
                    value="Mysterious",
                    label="Select Reading Atmosphere"
                )

            with gr.Tab("💭 Sentiment Filter"):
                use_sentiment = gr.Checkbox(label="Apply sentiment filter", value=False)
                sentiment_preference = gr.Dropdown(
                    ["Positive", "Negative", "Neutral"],
                    value="Positive",
                    label="Select Sentiment Preference"
                )

            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                **How it works:**
                - Enter a description of the book you want
                - Optionally filter by genre, mood, emotion, theme, atmosphere, and sentiment
                - AI combines semantic search with zero-shot classification for better matches
                - Results are ranked by both semantic similarity and classification relevance

                **New Features:**
                - **Zero-shot Classification**: Automatically categorizes books by emotion, theme, genre, and atmosphere
                - **Sentiment Analysis**: Analyzes emotional tone (positive/negative/neutral)
                - **Enhanced Ranking**: Combines semantic similarity with classification matching
                - **Batch Processing**: Efficiently processes thousands of books

                **Technical details:**
                - Uses precomputed Sentence Transformers all-MiniLM-L6-v2 embeddings
                - Zero-shot classification with Hugging Face transformers
                - Sentiment analysis with RoBERTa model
                - ChromaDB vector database for fast similarity search
                - LangChain for retrieval and processing
                - No API keys or internet connection required after setup
                """)

        with gr.Row():
            clear = gr.Button("🗑️ Clear", variant="secondary")
            run = gr.Button("🔍 Search Books", variant="primary")

        output = gr.Textbox(
            label="📚 Recommended Books",
            lines=20,
            placeholder="Your recommendations will appear here..."
        )

        run.click(
            recommend,
            inputs=[query, use_genre, use_mood, genre, mood, use_emotion, emotion, use_theme, theme, use_atmosphere, atmosphere, use_sentiment, sentiment_preference],
            outputs=output
        )

        clear.click(lambda: "", None, output)

    return demo