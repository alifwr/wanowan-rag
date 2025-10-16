from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from services.elastic_search import ElasticsearchService
from utils import chunk_by_feedback_entries, chunk_by_categories, chunk_semantic_with_overlap

if __name__ == "__main__":
    es_url = os.getenv("ES_HOST", "")
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="semantic-chunks-index",
        text_field="text",
        dense_vector_field="embeddings",
        num_characters_field="num_characters",
    )

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

    print(f"Loaded large document with {len(large_document)} characters")

    print("\nChoose chunking strategy:")
    print("1. Split by individual feedback entries")
    print("2. Group by feedback categories")
    print("3. Semantic chunking with sentence awareness")

    strategy = input("Enter strategy number (1-3): ").strip()

    if strategy == "1":
        chunks = chunk_by_feedback_entries(large_document)
        strategy_name = "feedback entries"
    elif strategy == "2":
        chunks = chunk_by_categories(large_document)
        strategy_name = "categories"
    elif strategy == "3":
        chunks = chunk_semantic_with_overlap(large_document)
        strategy_name = "semantic with overlap"
    else:
        print("Invalid choice, using default: feedback entries")
        chunks = chunk_by_feedback_entries(large_document)
        strategy_name = "feedback entries"

    print(f"\nUsing {strategy_name} strategy")
    print(f"Document split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"Preview: {chunk[:200]}...")

    num_indexed = es_service.index_data(
        index_name=es_service.index_name,
        text_field=es_service.text_field,
        dense_vector_field=es_service.dense_vector_field,
        embeddings=es_service.embeddings,
        texts=chunks,
        recreate_index=True,
    )

    print(f"\nSuccessfully indexed {num_indexed} chunks using {strategy_name} strategy")