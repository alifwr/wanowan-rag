from dotenv import load_dotenv

load_dotenv()
import os
from pathlib import Path
from services.elastic_search import ElasticsearchService

if __name__ == "__main__":
    es_url = os.getenv("ES_HOST", "")
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",
        text_field="text",
        dense_vector_field="embeddings",
        num_characters_field="num_characters",
    )

    docs_dir = Path("docs/chunked_docs")

    texts = []
    for file_path in docs_dir.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not texts:
        print("No documents found to index.")
        exit(1)

    print(f"Found {len(texts)} documents to index.")

    num_indexed = es_service.index_data(
        es_service.index_name,
        es_service.text_field,
        es_service.dense_vector_field,
        es_service.embeddings,
        texts,
        recreate_index=True,
    )
    print(f"Successfully indexed {num_indexed} documents.")
