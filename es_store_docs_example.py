"""
Document Storage Example for RAG Systems

This example demonstrates how to store pre-chunked documents in Elasticsearch
for retrieval-augmented generation (RAG) applications. Unlike the text storage
example that chunks large documents, this script works with documents that are
already appropriately sized for indexing.

Key concepts demonstrated:
- Loading environment variables for configuration
- Reading multiple text files from a directory
- Using ElasticsearchService for vector storage
- Indexing pre-processed documents with embeddings

Use case: When you have documents that are already chunked or naturally
small enough to be indexed as individual units (e.g., feedback entries,
short articles, or pre-processed chunks).
"""

from dotenv import load_dotenv

# Load environment variables from .env file (contains API keys, URLs, etc.)
load_dotenv()
import os
from pathlib import Path
from services.elastic_search import ElasticsearchService

if __name__ == "__main__":
    # Get Elasticsearch URL from environment variables
    # Falls back to empty string if ES_HOST not set
    es_url = os.getenv("ES_HOST", "")

    # Initialize Elasticsearch service for document storage
    # This creates a connection to Elasticsearch and sets up the index structure
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",  # Name of the Elasticsearch index
        text_field="text",              # Field name for storing document text
        dense_vector_field="embeddings", # Field name for storing vector embeddings
        num_characters_field="num_characters",  # Field for document length metadata
    )

    # Directory containing the pre-chunked documents
    # These are individual text files that will be indexed as separate documents
    docs_dir = Path("docs/chunked_docs")

    # List to store all document texts
    texts = []

    # Read all text files from the chunked_docs directory
    # Each file becomes one document in the vector database
    for file_path in docs_dir.glob("*.txt"):
        try:
            # Open and read each text file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()  # Remove leading/trailing whitespace
                if content:  # Only add non-empty documents
                    texts.append(content)
        except Exception as e:
            # Print error message if file reading fails
            print(f"Error reading {file_path}: {e}")

    # Check if any documents were found
    if not texts:
        print("No documents found to index.")
        exit(1)

    # Display the number of documents found
    print(f"Found {len(texts)} documents to index.")

    # Index all documents in Elasticsearch with vector embeddings
    # This process:
    # 1. Converts each text to vector embeddings using OpenAI
    # 2. Stores both text and embeddings in Elasticsearch
    # 3. Recreates the index if it already exists (recreate_index=True)
    num_indexed = es_service.index_data(
        es_service.index_name,
        es_service.text_field,
        es_service.dense_vector_field,
        es_service.embeddings,
        texts,
        recreate_index=True,
    )

    # Report successful indexing
    print(f"Successfully indexed {num_indexed} documents.")
