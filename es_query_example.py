"""
Query Example for RAG System

This example demonstrates how to search through your indexed documents using
vector similarity search. Instead of keyword matching, this uses semantic
similarity to find documents that are conceptually related to your query.

Key Concepts:
- Vector Search: Finding documents by meaning, not just keywords
- Semantic Similarity: "Customer complaints" might match "user feedback issues"
- Top-K Results: Return the most relevant documents
- Candidates: Number of documents to evaluate during search

Prerequisites:
- Run es_store_texts_example.py first to index some documents
- Or run es_store_large_docs_example.py for chunked documents
"""

from dotenv import load_dotenv

# Load environment variables (API keys, database connections, etc.)
load_dotenv()
import os
from services.elastic_search import ElasticsearchService

if __name__ == "__main__":
    # Get Elasticsearch connection URL from environment
    es_url = os.getenv("ES_HOST", "")

    # Initialize Elasticsearch service with same configuration as indexing
    # This ensures we're searching the same index where documents were stored
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",      # Must match the index used for storing
        text_field="text",                  # Field containing the text content
        dense_vector_field="embeddings",    # Field containing vector embeddings
        num_characters_field="num_characters",  # Metadata field
    )

    # Define your search query
    # This can be a question, topic, or any text you want to find related documents for
    query = "What is the most common feedbacks?"

    # Perform vector similarity search
    # top_k=3: Return top 3 most similar documents
    # candidates=5: Evaluate 5 candidate documents during search (for efficiency)
    results = es_service.query(query, top_k=3, candidates=5)

    # Display the results
    # Each result is a Document object with page_content (the text) and metadata
    for i, doc in enumerate(results):
        # Visual separator for readability
        print(
            "========================================================================================================================"
        )
        # Show result number and content
        print(f"Result {i+1}: {doc.page_content}")
        print(
            "========================================================================================================================"
        )
        print("\n")

    # Try different queries to see how vector search works:
    # - "mobile app problems" -> Should find app-related feedback
    # - "shipping issues" -> Should find delivery complaints
    # - "good customer service" -> Should find positive reviews
    # - "website improvements" -> Should find UX suggestions

    # Experiment with different parameters:
    # - Increase top_k for more results
    # - Increase candidates for potentially better results (but slower)
    # - Try more specific or general queries
