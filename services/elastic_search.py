from typing import List, Dict, Iterable
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_openai import OpenAIEmbeddings


class ElasticsearchService:
    def __init__(self, es_url: str, index_name: str, text_field: str, dense_vector_field: str, num_characters_field: str):
        self.es_url = es_url
        self.es_client = Elasticsearch([self.es_url])
        self.es_client.info()

        self.embeddings = OpenAIEmbeddings()

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
        if recreate and self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)
        
        try:
            self.es_client.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        text_field: {"type": "text"},
                        dense_vector_field: {"type": "dense_vector"},
                        num_characters_field: {"type": "integer"},
                    }
                },
            )
        except Exception as e:
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
        self.create_index(
            index_name, text_field, dense_vector_field, self.num_characters_field, recreate=recreate_index
        )

        vectors = embeddings.embed_documents(list(texts))
        requests = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": i,
                text_field: text,
                dense_vector_field: vector,
                self.num_characters_field: len(text),
            }
            for i, (text, vector) in enumerate(zip(texts, vectors))
        ]

        bulk(self.es_client, requests)

        if refresh:
            self.es_client.indices.refresh(index=index_name)

        return len(requests)
    
    def vector_query(self, query_search: str, top_k: int = 5, candidates: int = 10) -> Dict:
        vector = self.embeddings.embed_query(query_search)
        return {
            "knn": {
                "field": self.dense_vector_field,
                "query_vector": vector,
                "k": top_k,
                "num_candidates": candidates,
            }
        }

    def query(self, query_search: str, top_k: int = 5, candidates: int = 10) -> List[Document]:
        vector_retriever = ElasticsearchRetriever.from_es_params(
            index_name=self.index_name,
            body_func=lambda q: self.vector_query(q, top_k, candidates),
            content_field=self.text_field,
            url=self.es_url
        )

        return vector_retriever.invoke(query_search)