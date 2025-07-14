"""Qdrant client setup for RAG system."""

import os
from typing import Optional
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# Qdrant client
from qdrant_client import QdrantClient

load_dotenv()


def setup_qdrant_client() -> QdrantClient:
    """Setup Qdrant client connection."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    
    print(f"Connected to Qdrant at {qdrant_url}")
    return client


def setup_vector_store(
    collection_name: str = "medmax_pubmed",
    client: Optional[QdrantClient] = None
) -> QdrantVectorStore:
    """Setup LlamaIndex Qdrant vector store."""
    if client is None:
        client = setup_qdrant_client()
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            raise ValueError(f"Collection '{collection_name}' not found. Available: {collection_names}")
        
        print(f"📦 Using collection: {collection_name}")
        
    except Exception as e:
        raise ConnectionError(f"Failed to verify collection: {e}")
    
    # Create LlamaIndex vector store
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
    )
    
    print(f"Vector store configured for collection: {collection_name}")
    return vector_store


def setup_rag_client(collection_name: str = "medmax_pubmed") -> tuple[QdrantVectorStore, VectorStoreIndex]:
    """Setup complete RAG client with vector store and index."""
    # Setup vector store
    vector_store = setup_vector_store(collection_name)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create vector store index (loads existing data from Qdrant)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    
    print(f"RAG client ready with {collection_name}")
    return vector_store, index
