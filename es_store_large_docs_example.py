from dotenv import load_dotenv

load_dotenv()
import os
from pathlib import Path
from services.elastic_search import ElasticsearchService
from utils import chunk_text

if __name__ == "__main__":
    es_url = os.getenv("ES_HOST", "")
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",
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

    chunk_size = 800
    overlap = 100

    print(f"Chunking parameters: size={chunk_size}, overlap={overlap}")

    chunks = chunk_text(large_document, chunk_size, overlap)

    print(f"Document split into {len(chunks)} chunks")

    num_indexed = es_service.index_data(
        index_name=es_service.index_name,
        text_field=es_service.text_field,
        dense_vector_field=es_service.dense_vector_field,
        embeddings=es_service.embeddings,
        texts=chunks,
        recreate_index=True,
    )

    print(f"Successfully indexed {num_indexed} chunks")

    print("\nChunking details:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} characters")
        if i < 2:
            print(f"    Preview: {chunk[:100]}...")
        elif i == len(chunks) - 1:
            print(f"    Preview: {chunk[:100]}...")
