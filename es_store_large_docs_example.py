"""
Large Document Storage Example with Basic Chunking

This example demonstrates how to handle large documents by breaking them into
smaller chunks before indexing. This is essential when documents exceed the
token limits of language models or when you want more precise search results.

Chunking Strategy Used:
- Character-based chunking with intelligent boundary detection
- 800 characters per chunk with 100 character overlap
- Tries to break at sentence endings, line breaks, or word boundaries

Use Case:
- Processing long articles, reports, or books
- Breaking down large documents for better retrieval granularity
- Handling documents that are too big for direct embedding

The example loads a single large document and chunks it for storage.
"""

from dotenv import load_dotenv

# Load environment variables (API keys, database URLs, etc.)
load_dotenv()
import os
from pathlib import Path
from services.elastic_search import ElasticsearchService
from utils import chunk_text

if __name__ == "__main__":
    # Get Elasticsearch URL from environment variables
    es_url = os.getenv("ES_HOST", "")

    # Initialize Elasticsearch service
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",      # Index for storing chunked documents
        text_field="text",                  # Field for text content
        dense_vector_field="embeddings",    # Field for vector embeddings
        num_characters_field="num_characters",  # Field for chunk length
    )

    # Load the large document from file
    large_doc_path = Path("docs/large_doc.txt")

    try:
        with open(large_doc_path, "r", encoding="utf-8") as f:
            large_document = f.read().strip()
    except Exception as e:
        print(f"Error reading {large_doc_path}: {e}")
        exit(1)

    if not large_document:
        print("Large document is empty.")
        exit(1)

    # Display document statistics
    print(f"Loaded large document with {len(large_document)} characters")

    # Configure chunking parameters
    # These control how the document is split into smaller pieces
    chunk_size = 800  # Maximum characters per chunk
    overlap = 100     # Characters to overlap between chunks (maintains context)

    print(f"Chunking parameters: size={chunk_size}, overlap={overlap}")

    # Apply chunking to break the large document into smaller pieces
    chunks = chunk_text(large_document, chunk_size, overlap)

    print(f"Document split into {len(chunks)} chunks")

    # Index all chunks in Elasticsearch
    # Each chunk becomes a separate searchable document
    num_indexed = es_service.index_data(
        index_name=es_service.index_name,
        text_field=es_service.text_field,
        dense_vector_field=es_service.dense_vector_field,
        embeddings=es_service.embeddings,
        texts=chunks,  # List of chunks to index
        recreate_index=True,  # Replace any existing data
    )

    print(f"Successfully indexed {num_indexed} chunks")

    # Show chunking details for transparency
    print("\nChunking details:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} characters")
        if i < 2:  # Show preview of first 2 chunks
            print(f"    Preview: {chunk[:100]}...")
        elif i == len(chunks) - 1:  # Show preview of last chunk
            print(f"    Preview: {chunk[:100]}...")

    # Understanding the results:
    # - Each chunk is now a separate document in your search index
    # - Searches will return individual chunks, not the whole document
    # - Overlap ensures context continuity between chunks
    # - Try es_query_example.py to search through these chunks
