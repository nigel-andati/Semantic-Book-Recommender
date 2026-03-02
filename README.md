# Semantic Book Recommender

## Project Overview and Implementation

This project develops a semantic book recommendation system using 7,000 books from a Kaggle dataset, converting them into 384-dimensional embeddings with Sentence Transformers (all-MiniLM-L6-v2) and storing them in ChromaDB. The project aims to shift from keyword-restricted indexing to semantic natural language processing. Users can query in natural language to find books based on their intentions, utilizing Hugging Face transformers for real-time thematic and emotional analysis. An interactive Gradio web app allows users to filter recommendations by genre, mood, or theme. The script create_embeddings_subset.py generated the dataset embeddings before uploading to Google Drive.

### Data Pipeline:
1. Input: Natural language book descriptions
2. Output: Ranked book recommendations with relevance scores and metadata

## Running Instructions

1. Open the Colab notebook: Nigel_Andati_final_project_colab.ipynb.
2. Click Runtime then Run all
3. Wait ~3 minutes for automatic setup (downloads ~15MB embeddings from Google Drive)
4. Access the interactive interface via the generated public URL at the bottom of the notebook

