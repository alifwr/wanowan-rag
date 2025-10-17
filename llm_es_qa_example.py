"""
Question Answering Example with RAG System

This example demonstrates how to build a complete Question-Answering system using
Retrieval-Augmented Generation (RAG). It combines document retrieval from Elasticsearch
with a Large Language Model to provide accurate, context-aware answers.

Key Concepts:
- RetrievalQA: Combines retrieval and generation for QA tasks
- Context Retrieval: Fetches relevant documents before answering
- LLM Integration: Uses OpenAI's language models for answer generation
- Conversational Memory: Maintains context across multiple questions

Prerequisites:
- Run es_store_texts_example.py first to index some documents
- Or run es_store_large_docs_example.py for chunked documents
- OpenAI API key configured in environment variables

The system will:
1. Take a question from the user
2. Retrieve relevant documents from Elasticsearch
3. Use the LLM to generate an answer based on the retrieved context
"""

from dotenv import load_dotenv

# Load environment variables (API keys, database connections, etc.)
load_dotenv()
import os
from services.elastic_search import ElasticsearchService
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    # Get Elasticsearch connection URL from environment
    es_url = os.getenv("ES_HOST", "")

    # Initialize Elasticsearch service with same configuration as indexing
    # This ensures we're using the same index where documents were stored
    es_service = ElasticsearchService(
        es_url=es_url,
        index_name="large-docs-index",      # Must match the index used for storing
        text_field="text",                  # Field containing the text content
        dense_vector_field="embeddings",    # Field containing vector embeddings
        num_characters_field="num_characters",  # Metadata field
    )

    # Create a retriever from the Elasticsearch service
    # This retriever knows how to search our vector index and return relevant documents
    retriever = es_service.get_retriever(top_k=3, candidates=5)

    # Initialize the language model for answer generation
    # Using GPT-3.5-turbo for cost-effective QA, but GPT-4 could provide better answers
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",      # You can change to "gpt-4" for better quality
        temperature=0.1,             # Low temperature for consistent, factual answers
    )

    # Create the RetrievalQA chain
    # This combines retrieval (from Elasticsearch) with generation (from LLM)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                    # The language model to use for answering
        chain_type="stuff",         # "stuff" puts all retrieved docs into context
        retriever=retriever,        # The retriever that finds relevant documents
        return_source_documents=True,  # Include the source docs in the response
    )

    # Example questions to demonstrate the QA system
    # These questions are designed to test retrieval from the indexed documents
    questions = [
        "What are the main customer complaints about the mobile app?",
        "How has customer service improved according to the feedback?",
        "What issues do customers have with product images?",
        "Can you summarize the common payment and checkout problems?",
    ]

    print("🤖 RAG Question-Answering System")
    print("=" * 50)
    print("This system answers questions based on the documents in your knowledge base.\n")

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 60)

        try:
            # Get the answer from the QA chain
            # This will retrieve relevant documents and generate an answer
            result = qa_chain.invoke({"query": question})

            # Display the answer
            print(f"Answer: {result['result']}")
            print()

            # Show which source documents were used
            print("Source Documents Used:")
            for j, doc in enumerate(result['source_documents'], 1):
                # Truncate long documents for display
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                print(f"  {j}. {content_preview}")
            print()

        except Exception as e:
            print(f"Error processing question: {e}")
            print()

        print("=" * 60)
        print()

    # Interactive mode - allow user to ask their own questions
    print("💬 Interactive Mode")
    print("Enter your own questions (type 'quit' to exit):")
    print("-" * 50)

    while True:
        try:
            user_question = input("\nYour question: ").strip()

            if user_question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break

            if not user_question:
                continue

            # Process the user's question
            result = qa_chain.invoke({"query": user_question})

            print(f"\nAnswer: {result['result']}")
            print("\nSource documents used:")
            for j, doc in enumerate(result['source_documents'], 1):
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  {j}. {content_preview}")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
