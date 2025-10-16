"""
Elasticsearch Service for RAG (Retrieval-Augmented Generation) System

This module provides a high-level interface to Elasticsearch for storing and retrieving
text documents with vector embeddings. It's specifically designed for RAG applications
where you need to:

1. Store documents with their semantic embeddings (vector representations)
2. Perform similarity search using vector queries
3. Retrieve relevant documents for question-answering

The service uses:
- Elasticsearch for document storage and search
- OpenAI embeddings for converting text to vectors
- LangChain for vector retrieval operations

Key Concepts:
- Index: Like a database table, stores documents with the same structure
- Document: Individual records containing text, embeddings, and metadata
- Vector Search: Finding similar documents using mathematical similarity of embeddings
"""

from typing import List, Dict, Iterable
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_openai import OpenAIEmbeddings


class ElasticsearchService:
    """
    Service class for managing Elasticsearch operations in a RAG system.

    This class handles all interactions with Elasticsearch, including:
    - Creating and managing indices
    - Storing documents with embeddings
    - Performing vector similarity searches
    - Retrieving relevant documents for queries

    Attributes:
        es_url (str): URL of the Elasticsearch server
        es_client: Elasticsearch client instance
        embeddings: OpenAI embeddings model for text-to-vector conversion
        index_name (str): Name of the Elasticsearch index
        text_field (str): Field name for storing text content
        dense_vector_field (str): Field name for storing vector embeddings
        num_characters_field (str): Field name for storing text length metadata
    """

    def __init__(self, es_url: str, index_name: str, text_field: str, dense_vector_field: str, num_characters_field: str):
        """
        Initialize the Elasticsearch service.

        Args:
            es_url (str): Elasticsearch server URL (e.g., "http://localhost:9200")
            index_name (str): Name for the Elasticsearch index
            text_field (str): Field name for text content (e.g., "text")
            dense_vector_field (str): Field name for embeddings (e.g., "embeddings")
            num_characters_field (str): Field name for character count (e.g., "num_characters")
        """
        # Store configuration
        self.es_url = es_url

        # Create Elasticsearch client and test connection
        self.es_client = Elasticsearch([self.es_url])
        self.es_client.info()  # This will raise an exception if connection fails

        # Initialize embeddings model (converts text to vectors)
        self.embeddings = OpenAIEmbeddings()

        # Store field names for the index mapping
        self.index_name = index_name
        self.text_field = text_field
        self.dense_vector_field = dense_vector_field
        self.num_characters_field = num_characters_field

    def create_index(
        self,
        index_name: str,
        text_field: str,
        dense_vector_field: str,
        num_characters_field: str,
        recreate: bool = False,
    ):
        """
        Create an Elasticsearch index with the proper mapping for text and vectors.

        An index in Elasticsearch is like a database table. It defines the structure
        (mapping) of documents that will be stored. For RAG, we need:

        1. A text field for storing the actual content
        2. A dense_vector field for storing embeddings (vector representations)
        3. An integer field for storing text length (useful for filtering)

        Args:
            index_name (str): Name of the index to create
            text_field (str): Field name for text content
            dense_vector_field (str): Field name for vector embeddings
            num_characters_field (str): Field name for character count
            recreate (bool): If True, delete existing index before creating new one

        Raises:
            Exception: If index creation fails (except for "already exists" errors)
        """
        # If recreate is True and index exists, delete it first
        if recreate and self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)

        try:
            # Create the index with proper mapping
            self.es_client.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        # Text field for full-text search and retrieval
                        text_field: {"type": "text"},
                        # Dense vector field for similarity search with embeddings
                        dense_vector_field: {"type": "dense_vector"},
                        # Integer field for storing text length
                        num_characters_field: {"type": "integer"},
                    }
                },
            )
        except Exception as e:
            # Only ignore "resource_already_exists_exception" (index already exists)
            # Re-raise other exceptions as they indicate real problems
            if "resource_already_exists_exception" not in str(e):
                raise

    def index_data(
        self,
        index_name: str,
        text_field: str,
        dense_vector_field: str,
        embeddings: Embeddings,
        texts: Iterable[str],
        refresh: bool = True,
        recreate_index: bool = False,
    ):
        """
        Index multiple text documents with their vector embeddings.

        This method takes a collection of text documents, converts them to vector
        embeddings, and stores them in Elasticsearch for later retrieval.

        Process:
        1. Ensure the index exists (create if needed)
        2. Convert all texts to vector embeddings using the embeddings model
        3. Prepare bulk indexing requests with text, vectors, and metadata
        4. Execute bulk indexing operation
        5. Refresh the index to make documents immediately searchable

        Args:
            index_name (str): Name of the index to store documents in
            text_field (str): Field name for storing text content
            dense_vector_field (str): Field name for storing embeddings
            embeddings (Embeddings): Model to convert text to vectors
            texts (Iterable[str]): Collection of text documents to index
            refresh (bool): Whether to refresh index after indexing (default: True)
            recreate_index (bool): Whether to recreate index if it exists (default: False)

        Returns:
            int: Number of documents successfully indexed
        """
        # Ensure the index exists with proper mapping
        self.create_index(
            index_name, text_field, dense_vector_field, self.num_characters_field, recreate=recreate_index
        )

        # Convert all texts to vector embeddings in batch
        # This is more efficient than converting one at a time
        texts_list = list(texts)
        vectors = embeddings.embed_documents(texts_list)

        # Prepare bulk indexing requests
        # Each document gets: text content, vector embedding, character count, and unique ID
        requests = [
            {
                "_op_type": "index",        # Operation type: index (add/update)
                "_index": index_name,       # Which index to store in
                "_id": i,                   # Unique document ID (sequential)
                text_field: text,           # The actual text content
                dense_vector_field: vector, # Vector embedding for similarity search
                self.num_characters_field: len(text),  # Text length for filtering
            }
            for i, (text, vector) in enumerate(zip(texts_list, vectors))
        ]

        # Execute bulk indexing operation
        # This is much faster than indexing documents one by one
        bulk(self.es_client, requests)

        # Refresh the index to make documents immediately available for search
        if refresh:
            self.es_client.indices.refresh(index=index_name)

        # Return the number of documents indexed
        return len(requests)
    
    def vector_query(self, query_search: str, top_k: int = 5, candidates: int = 10) -> Dict:
        """
        Create a vector similarity search query for Elasticsearch.

        This method converts a text query into a vector embedding and creates
        an Elasticsearch KNN (k-nearest neighbors) query to find similar documents.

        KNN Search Process:
        1. Convert query text to vector embedding
        2. Search for documents with most similar vectors
        3. Return top_k most similar documents

        Args:
            query_search (str): The search query text
            top_k (int): Number of most similar documents to return (default: 5)
            candidates (int): Number of candidates to consider during search (default: 10)

        Returns:
            Dict: Elasticsearch query dictionary with KNN configuration
        """
        # Convert the search query to a vector embedding
        vector = self.embeddings.embed_query(query_search)

        # Create KNN (k-nearest neighbors) query
        return {
            "knn": {
                "field": self.dense_vector_field,  # Which field contains the vectors
                "query_vector": vector,            # The query vector to search with
                "k": top_k,                        # How many results to return
                "num_candidates": candidates,      # How many candidates to evaluate
            }
        }

    def query(self, query_search: str, top_k: int = 5, candidates: int = 10) -> List[Document]:
        """
        Perform a vector similarity search and return relevant documents.

        This is the main method for searching your document collection. It uses
        vector similarity to find documents that are semantically similar to
        your query, rather than just keyword matching.

        The method uses LangChain's ElasticsearchRetriever which handles:
        - Converting queries to vectors
        - Executing KNN search in Elasticsearch
        - Returning results as Document objects

        Args:
            query_search (str): The search query (question or topic)
            top_k (int): Number of top results to return (default: 5)
            candidates (int): Number of candidates to evaluate (default: 10)

        Returns:
            List[Document]: List of relevant documents with their content and metadata

        Example:
            >>> results = es_service.query("How do I reset my password?", top_k=3)
            >>> for doc in results:
            ...     print(doc.page_content)  # The text content
            ...     print(doc.metadata)      # Additional metadata
        """
        # Create a retriever that knows how to search our index
        # body_func is a function that generates the search query for a given input
        vector_retriever = ElasticsearchRetriever.from_es_params(
            index_name=self.index_name,                    # Which index to search
            body_func=lambda q: self.vector_query(q, top_k, candidates),  # How to create search queries
            content_field=self.text_field,                 # Which field contains the text
            url=self.es_url                               # Elasticsearch server URL
        )

        # Execute the search and return results
        return vector_retriever.invoke(query_search)