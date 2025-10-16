"""
Text Storage Example for RAG System

This example demonstrates how to store a collection of text documents in Elasticsearch
with vector embeddings for later retrieval. This is the simplest form of document
storage where each text snippet becomes a separate searchable document.

Use Case:
- Storing individual sentences, reviews, or short text snippets
- Creating a knowledge base from text fragments
- Testing the basic indexing and search functionality

The example uses e-commerce customer feedback texts, but you can replace them
with any collection of texts relevant to your use case.
"""

from dotenv import load_dotenv

# Load environment variables from .env file (contains API keys, database URLs, etc.)
load_dotenv()
import os
from services.elastic_search import ElasticsearchService

if __name__ == "__main__":
    # Get Elasticsearch URL from environment variables
    # This allows different configurations for development/production
    es_url = os.getenv("ES_HOST", "")

    # Initialize the Elasticsearch service with configuration
    # This creates a connection to Elasticsearch and sets up the embeddings model
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",      # Name of the index (like a table name)
        text_field="text",                  # Field name for storing text content
        dense_vector_field="embeddings",    # Field name for storing vector embeddings
        num_characters_field="num_characters",  # Field for text length metadata
    )

    # Collection of text documents to index
    # Each string becomes a separate document in the search index
    # These are example e-commerce customer feedback statements
    texts = [
        "The mobile app crashes frequently when adding items to cart during peak hours.",
        "Customer service response time has improved significantly since the last update.",
        "The new search filter feature makes finding products much easier and faster.",
        "Package tracking information is often inaccurate or delayed by several days.",
        "The website's mobile responsiveness needs major improvements for better user experience.",
        "Payment processing is secure but the checkout flow could be simplified.",
        "Product images don't always match the actual items received by customers.",
        "The loyalty program offers great rewards but the redemption process is confusing.",
        "Shipping costs are too high compared to competitors in the same price range.",
        "The return policy is clear but the actual return process takes too long.",
        "Product descriptions are detailed but sometimes contain inaccurate information.",
        "The wishlist feature works well but could use better organization options.",
        "Email notifications for order updates are helpful but sometimes arrive late.",
        "The size guide for clothing items is not accurate enough for consistent sizing.",
        "Customer reviews are helpful but the filtering and sorting options could be improved.",
        "The discount codes work reliably but expiration dates are confusing.",
        "Live chat support is available but response times vary greatly by time of day.",
        "The product recommendation algorithm suggests relevant items most of the time.",
        "Gift wrapping options are limited and the quality could be better.",
        "Account security features are robust but two-factor authentication setup is complex.",
        "The mobile app's push notifications are useful but can be overwhelming.",
        "Product availability indicators are not always accurate in real-time.",
        "The checkout process saves payment information securely for future purchases.",
        "Customer feedback forms are easy to find but the submission process is lengthy.",
        "The website's loading speed has improved but still lags during sales events.",
        "Product categories are well-organized but subcategories could be more intuitive.",
    ]

    # Index all the text documents
    # This converts texts to embeddings and stores them in Elasticsearch
    # recreate_index=True means it will replace any existing data
    num_indexed = es_service.index_data(
        index_name=es_service.index_name,
        text_field=es_service.text_field,
        dense_vector_field=es_service.dense_vector_field,
        embeddings=es_service.embeddings,
        texts=texts,
        recreate_index=True,  # Replace existing index if it exists
    )

    # Report the results
    print(f"Indexed {num_indexed} documents.")

    # Next steps:
    # 1. Run es_query_example.py to search through these documents
    # 2. Try different queries to see how vector search works
    # 3. Experiment with different top_k and candidates values
