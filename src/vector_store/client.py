"""Qdrant client setup and collection management."""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()


def setup_qdrant_client() -> QdrantClient:
    """Setup Qdrant client connection."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        return QdrantClient(url=qdrant_url)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if collection exists in Qdrant."""
    try:
        collections = client.get_collections().collections
        collections_names = [col.name for col in collections]
        return collection_name in collections_names
    except Exception as e:
        print(f"Error checking collections: {e}")
        return False


def collection_has_data(client: QdrantClient, collection_name: str) -> bool:
    """Check if collection has data (vectors) to avoid unnecessary reindexing."""
    try:
        if not collection_exists(client, collection_name):
            return False
            
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count
        print(f"Collection '{collection_name}' has {points_count} points.")
        return points_count > 0
    except Exception as e:
        print(f"Error checking collection data: {e}")
        return False


def create_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Create a new collection in Qdrant."""
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Ensure the collection exists, create if it doesn't."""
    if not collection_exists(client, collection_name):
        create_collection(client, collection_name, vector_size)
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")
