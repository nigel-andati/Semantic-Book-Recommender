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

## Bonus Contribution

### Exceptional Original Design
**Novel System Architecture**: This project introduces an original semantic search architecture that combines Sentence Transformers embeddings with zero-shot classification to understand both lexical and emotional dimensions of book recommendations. Unlike traditional keyword-based systems, the architecture captures semantic meaning and emotional intent through a two-tier ranking system: (1) vector similarity for content matching and (2) classification-based filtering for thematic/emotional alignment.

**Reproducibility-First Design**: The system was architected from the ground up to prioritize academic reproducibility over performance shortcuts. This required rejecting initially faster but less reproducible approaches (OpenAI embeddings, Hugging Face API calls) in favor of local Sentence Transformers processing, despite the increased computational requirements.

### Advanced and Transparent AI Use
**Iterative AI-Guided Development**: The project demonstrates thoughtful AI integration through multiple development cycles. Initial AI-generated code was systematically tested, debugged, and restructured. For instance, the embedding generation script underwent complete architectural revision to make it standalone and cross-platform compatible.

**Controlled AI Experimentation**: Multiple AI models and approaches were experimentally evaluated. OpenAI embeddings were tested and rejected due to network instability; TF-IDF was considered but deemed insufficient for semantic understanding; the final Sentence Transformers choice resulted from empirical testing across different embedding dimensions and performance metrics.

### Substantive Human Revision of AI Output
**Complete System Restructuring**: AI-generated components underwent substantial human intervention beyond feature-level edits. The create_embeddings_subset.py script was transformed from a simple generation tool into a comprehensive, self-contained pipeline with embedded data loading, processing, and export functionality - representing a complete architectural redesign.

**Cross-Platform Optimization**: Significant human effort was invested in resolving Windows-specific Unicode encoding issues and Colab environment incompatibilities, requiring deep understanding of both platforms' constraints and implementing custom solutions rather than relying on AI-suggested workarounds.

### Reflective Understanding
**Empirical Evaluation of AI Limitations**: Through systematic testing, I identified critical limitations in AI-generated solutions. Network-dependent approaches failed unpredictably in different environments, and initial performance optimizations created fragile dependencies. This empirical evidence drove the decision toward local, deterministic processing despite higher computational costs.

**Conceptual Framework Development**: The core innovation - combining semantic embeddings with zero-shot classification - emerged from recognizing that semantic similarity alone was insufficient for meaningful book recommendations. This conceptual breakthrough required understanding both the technical capabilities of different AI approaches and the human factors of book discovery, going beyond what AI tools could suggest.

### Originality Demonstration
These contributions extend beyond baseline requirements by establishing a reproducible, academically sound methodology for semantic search that prioritizes reliability over convenience. The system's architecture demonstrates original intellectual input in balancing technical feasibility with scholarly rigor, creating a framework that can be confidently used in research and educational contexts.

The design choices reflect deep engagement with both the technical challenges of semantic search and the human-centered aspects of recommendation systems, resulting in a solution that is both technically sophisticated and practically deployable.
