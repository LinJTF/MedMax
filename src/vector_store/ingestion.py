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
        # Create searchable text content
        contexts_text = " ".join(record['contexts'])
        page_content = (
            f"Question: {record['question']}\n"
            f"Context: {contexts_text}\n"
            f"Answer: {record['final_decision']}\n"
            f"Detailed Explanation: {record['long_answer']}"
        )
        
        # Create payload with metadata
        payload = {
            "text": page_content,
            "metadata": {
                "question": record["question"],
                "contexts": record["contexts"],
                "final_decision": record["final_decision"],
                "long_answer": record["long_answer"],
                "record_id": idx
            },
        }
        
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
