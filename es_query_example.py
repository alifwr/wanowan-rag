from dotenv import load_dotenv

load_dotenv()
import os
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

    query = "What is the most common feedbacks?"
    results = es_service.query(query, top_k=3, candidates=5)
    for i, doc in enumerate(results):
        print(
            "========================================================================================================================"
        )
        print(f"Result {i+1}: {doc.page_content}")
        print(
            "========================================================================================================================"
        )
        print("\n")
