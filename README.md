# RAG (Retrieval-Augmented Generation) System with Elasticsearch

A comprehensive implementation of a Retrieval-Augmented Generation system using Elasticsearch as the vector database, OpenAI embeddings for semantic search, and LangChain for document processing. This codebase serves as both a functional RAG system and an educational resource for learning about vector search, document chunking, and semantic similarity.

## 🚀 Features

- **Vector Search**: Semantic similarity search using OpenAI embeddings
- **Multiple Chunking Strategies**: Character-based, semantic, and custom chunking methods
- **Elasticsearch Integration**: High-performance vector storage and retrieval
- **Docker Deployment**: Easy setup with Docker Compose
- **Educational Comments**: Comprehensive documentation for beginners
- **Modular Architecture**: Clean separation of concerns

## 📋 Prerequisites

- Python 3.12 or higher
- Docker and Docker Compose
- OpenAI API key
- Git

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/alifwr/wanowan-rag.git
cd wanowan-rag
```

### 2. Set Up Python Environment

```bash
# Install dependencies using uv (recommended)
uv sync

# Or create virtual environment manually and use uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Note**: This project uses `uv.lock` for reproducible dependency management. The `uv sync` command will install all dependencies as specified in the lock file.

### 3. Configure Environment Variables

Copy the `.env` file and update it with your OpenAI API key:

```bash
cp .env.example .env  # If you have an example file
# Or edit .env directly
```

Required environment variables:
- `ES_HOST`: Elasticsearch URL (default: `http://localhost:9200`)
- `OPENAI_API_KEY`: Your OpenAI API key

### 4. Start Elasticsearch with Docker Compose

```bash
# Start Elasticsearch and Kibana
docker-compose up -d

# Wait for Elasticsearch to be ready (this may take 1-2 minutes)
# You can check the status with:
curl -X GET "localhost:9200/_cluster/health?pretty"
```

### 5. Verify Setup

```bash
# Test Elasticsearch connection
python -c "from services.elastic_search import ElasticsearchService; es = ElasticsearchService('http://localhost:9200', 'test', 'text', 'embeddings', 'num_chars'); print('Connection successful!')"
```

## 📖 Understanding RAG and Vector Search

### What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances language models by retrieving relevant information from a knowledge base before generating responses. Instead of relying solely on the model's training data, RAG systems:

1. **Index Documents**: Store documents with their semantic embeddings (vector representations)
2. **Retrieve Context**: Find relevant documents using vector similarity search
3. **Generate Answers**: Use retrieved documents as context for the language model

### Vector Search vs. Keyword Search

- **Keyword Search**: Finds exact word matches ("cat" matches "cat")
- **Vector Search**: Finds semantic similarity ("feline" might match "cat" because they mean similar things)

Vectors (embeddings) capture the meaning of text, allowing searches like:
- "Customer complaints" → matches "user feedback issues"
- "Payment problems" → matches "billing difficulties"

## 🏗️ Architecture

```
├── services/
│   └── elastic_search.py    # Elasticsearch service layer
├── utils.py                 # Text chunking utilities
├── es_store_*.py           # Document indexing examples
├── es_query_example.py     # Search and retrieval example
├── docker-compose.yml      # Elasticsearch deployment
└── .env                    # Environment configuration
```

### Core Components

1. **ElasticsearchService**: Handles all Elasticsearch operations
   - Index creation and management
   - Document storage with embeddings
   - Vector similarity search
   - Integration with LangChain

2. **Text Chunking Utilities**: Multiple strategies for breaking large documents
   - Character-based chunking with overlap
   - Semantic chunking by content
   - Custom chunking for specific document types

## 📚 Examples and Use Cases

### 1. Basic Text Storage (`es_store_texts_example.py`)

**Purpose**: Store individual text snippets as separate documents.

**Use Case**: When you have short, self-contained pieces of text like:
- Customer reviews
- FAQ entries
- Short articles
- Individual sentences

**How it works**:
```python
texts = [
    "The mobile app crashes frequently...",
    "Customer service has improved...",
    # ... more texts
]

es_service.index_data(index_name, text_field, vector_field, embeddings, texts)
```

### 2. Large Document Processing (`es_store_large_docs_example.py`)

**Purpose**: Break large documents into smaller chunks for better search precision.

**Use Case**: When dealing with long documents like:
- Books or articles
- Research papers
- Long reports
- Documentation

**Chunking Strategy**:
- Chunk size: 800 characters
- Overlap: 100 characters
- Intelligent boundary detection (sentences, paragraphs)

### 3. Pre-chunked Documents (`es_store_docs_example.py`)

**Purpose**: Store documents that are already appropriately sized.

**Use Case**: When you have pre-processed documents like:
- Individual feedback entries
- Chapter sections
- Pre-segmented content

**How it works**: Reads text files from `docs/chunked_docs/` directory and indexes each as a separate document.

### 4. Query and Retrieval (`es_query_example.py`)

**Purpose**: Search through indexed documents using semantic similarity.

**Key Parameters**:
- `top_k`: Number of results to return (e.g., 3)
- `candidates`: Number of documents to evaluate (e.g., 5)

**Example Query**:
```python
query = "What is the most common feedback?"
results = es_service.query(query, top_k=3, candidates=5)
```

### 5. Question Answering with LLM (`llm_es_qa_example.py`)

**Purpose**: Build a complete QA system that retrieves documents and generates answers using a Large Language Model.

**Use Case**: When you want conversational, context-aware answers based on your document collection:
- Customer support chatbots
- Document Q&A systems
- Knowledge base assistants
- Research question answering

**Key Features**:
- Combines retrieval with generation
- Supports both batch and interactive questioning
- Provides source document attribution
- Uses GPT-3.5-turbo or GPT-4 for answer generation

**How it works**:
```python
# Create retriever and LLM
retriever = es_service.get_retriever(top_k=3, candidates=5)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask questions
result = qa_chain.invoke({"query": "What are the main customer complaints?"})
```

## 🔧 Chunking Strategies

**Current Implementation**: This project primarily uses simple character-based chunking with intelligent boundary detection. The examples demonstrate basic chunking strategies suitable for most use cases, but text chunking is an active area of research with many advanced techniques available.

**Advanced Chunking Strategies**: There are numerous sophisticated chunking methods beyond what's implemented here, including:
- Recursive text splitting with document structure awareness
- Agent-based chunking with language model guidance
- Context-aware splitting using embeddings
- Document-specific chunking (code, markdown, structured data)

For comprehensive information on advanced chunking strategies, see:
- [LangChain Text Splitters Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [LlamaIndex Chunking Guide](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [Advanced Chunking Techniques Research](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

### 1. Character-Based Chunking (`chunk_text`) - **Currently Implemented**

- Splits text at approximately `chunk_size` characters
- Adds `overlap` characters between chunks
- Tries to break at natural boundaries (sentences, words)

**Pros**: Simple, fast, works with any text
**Cons**: May split related concepts

### 2. Semantic Chunking (`chunk_semantic_with_overlap`)

- Uses regex patterns to identify content boundaries
- Recognizes headers, lists, and structural elements
- Maintains semantic coherence

**Pros**: Preserves meaning and context
**Cons**: Requires pattern tuning for different document types

### 3. Custom Chunking Methods

- `chunk_by_feedback_entries`: Specialized for feedback data
- `chunk_by_categories`: Groups content by categories
- Extensible for domain-specific needs

## 🚀 Running the Examples

### Step 1: Start Services
```bash
docker-compose up -d
```

### Step 2: Index Documents
Choose one of the storage examples:

```bash
# For short texts
python es_store_texts_example.py

# For large documents
python es_store_large_docs_example.py

# For pre-chunked documents
python es_store_docs_example.py
```

### Step 3: Query and Search
```bash
python es_query_example.py
```

### Step 4: Question Answering (Optional)
For a complete RAG experience with LLM-powered answers:
```bash
python llm_es_qa_example.py
```

## 🔍 Understanding the Results

When you run a query, you'll get results like:

```
Query: "What is the most common feedback?"

Results:
1. "The mobile app crashes frequently when adding items to cart during peak hours."
   (Similarity: 0.87)

2. "Customer service response time has improved significantly since the last update."
   (Similarity: 0.82)

3. "The website's mobile responsiveness needs major improvements for better user experience."
   (Similarity: 0.79)
```

The similarity scores indicate how semantically related each document is to your query.

## 🛠️ Customization

### Changing Chunk Sizes

Edit the chunking parameters in the examples:

```python
# In es_store_large_docs_example.py
chunks = chunk_text(content, chunk_size=800, overlap=100)
```

### Using Different Embedding Models

Modify the ElasticsearchService initialization:

```python
# Instead of OpenAI embeddings
from langchain_huggingface import HuggingFaceEmbeddings
self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Custom Index Configuration

Adjust the index mapping in `create_index()` method for different field types or analyzers.

## 📊 Monitoring and Management

### Kibana Dashboard

Access Kibana at `http://localhost:5601` to:
- Visualize your indices
- Monitor Elasticsearch performance
- Create custom dashboards
- Explore indexed documents

### Elasticsearch API

Direct API access for advanced operations:

```bash
# Check cluster health
curl -X GET "localhost:9200/_cluster/health?pretty"

# List all indices
curl -X GET "localhost:9200/_cat/indices?v"

# View index mapping
curl -X GET "localhost:9200/large-docs-index/_mapping?pretty"
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure Elasticsearch is running
   ```bash
   docker-compose ps
   docker-compose logs es01
   ```

2. **OpenAI API Errors**: Check your API key and quota
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

### Performance Tuning

- **Index Refresh**: Adjust refresh intervals for bulk indexing
- **Vector Dimensions**: Consider dimensionality reduction for large datasets
- **Query Optimization**: Use appropriate `candidates` values

## 📚 Learn More

### Key Concepts to Explore

1. **Embeddings**: How text becomes vectors
2. **Cosine Similarity**: The math behind vector search
3. **Index Optimization**: Making searches faster
4. **Hybrid Search**: Combining keyword and vector search

### Recommended Reading

- [Elasticsearch Vector Search Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

**Happy RAG-ing!** 🎉

This codebase demonstrates how to build production-ready RAG systems with modern vector search capabilities. Use it as a starting point for your own applications or as a learning resource for understanding semantic search and document processing.</content>
<parameter name="filePath">/home/alif/dd/rag/README.md
