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

## AI Use and External Resources Disclosure
### AI Tools and Prompts
• Grok (xAI) and Cursor/ Composer 1.5 
• Prompts used: 
  - “Create an interactive Gradio interface for the book recommender featuring text input for queries, sliders for result limits, dropdown filters for emotion/theme/genre, example queries, and formatted results showing book details with relevance scores and classifications.”
  - "Create a Python script to generate Sentence Transformers embeddings for a 7k book Kaggle dataset. Include data cleaning, genre/mood assignment via keywords, embedding generation, and ZIP file output."
  - “Resolve runtime errors during book addition to the ChromaDB database, add clear explanations and comments throughout the code, and optimize embedding queries.”

### AI-Assisted Development
AI was used to produce the create_embeddings_subset.py file and build the initial framework for the Gradio user interface. During development, I used the AI for debugging to identify and resolve configuration errors when integrating Google Drive as a cloud storage equivalent and when creating the search function. Additionally, the AI provided component integration support by supplying the code necessary to connect the vector database with the application frontend (Gradio).
### Self-Authored Sections
I used Google Drive for cloud storage to make the project faster and easy for others to run. I independently built the core design, including the semantic search and ranking systems, and manually wrote all the specific genres, moods, and themes used for filtering. My original innovation was combining Sentence Transformers with zero-shot classification to allow the system to understand the actual meaning and emotional intent behind a user's search. I also edited the Gradio UI once generated to improve output.

### External Resources
• Kaggle Dataset: "7k-books-with-metadata"
• LLM Course – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio) – freeCodeCamp.org (YouTube, 2025).
