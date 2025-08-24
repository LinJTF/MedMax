"""Data ingestion to Qdrant vector store."""

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


def upload_pubmed_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    pubmed_records: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Upload PubMed records and embeddings to Qdrant."""
    if len(pubmed_records) != len(embeddings):
        raise ValueError("Number of records must match number of embeddings")
    
    points = []
    batch_size = 100  # Process in batches to avoid memory issues
    
    for idx, (record, embedding) in enumerate(zip(pubmed_records, embeddings)):
        # Create searchable text content - ONLY the context part for RAG retrieval
        contexts_text = " ".join(record['contexts'])
        
        # Create payload with metadata (question, answer, etc. stored as metadata only)
        payload = {
            "text": contexts_text,  # Only context for RAG retrieval
            "metadata": {
                "question": record["question"],
                "contexts": record["contexts"],
                "long_answer": record["long_answer"],
                "record_id": idx
            },
        }
        
        # Add final_decision if available
        if 'final_decision' in record:
            payload["metadata"]["final_decision"] = record["final_decision"]
        
        # Add dataset source info if available
        if 'dataset_source' in record:
            payload["metadata"]["dataset_source"] = record["dataset_source"]
        if 'pubid' in record:
            payload["metadata"]["pubid"] = record["pubid"]
        
        points.append(PointStruct(id=idx, vector=embedding, payload=payload))
        
        # Upload in batches
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            print(f"Uploaded batch of {len(points)} points (total processed: {idx + 1})")
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded final batch of {len(points)} points")
    
    print(f"Successfully uploaded {len(pubmed_records)} PubMed records to collection '{collection_name}'")
