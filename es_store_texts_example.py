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

    num_indexed = es_service.index_data(
        es_service.index_name,
        es_service.text_field,
        es_service.dense_vector_field,
        es_service.embeddings,
        texts,
        recreate_index=True,
    )
    print(f"Indexed {num_indexed} documents.")
