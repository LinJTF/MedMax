"""Custom retriever for Qdrant integration."""

from typing import List, Optional, Any, Dict
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import Settings
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint


class QdrantRetriever(BaseRetriever):
    """Custom retriever that directly queries Qdrant for medical data."""
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialize Qdrant retriever."""
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        super().__init__(**kwargs)
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve relevant documents from Qdrant."""
        # Get embedding for query
        embed_model = Settings.embed_model
        query_embedding = embed_model.get_query_embedding(query_bundle.query_str)
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )
        
        # Convert to LlamaIndex nodes
        nodes_with_scores = []
        for result in search_results:
            if isinstance(result, ScoredPoint):
                # Extract metadata and text
                payload = result.payload or {}
                text_content = payload.get("text", "")
                metadata = payload.get("metadata", {})
                
                # Create node
                from llama_index.core.schema import TextNode
                node = TextNode(
                    text=text_content,
                    metadata=metadata,
                    id_=str(result.id)
                )
                
                # Create node with score
                node_with_score = NodeWithScore(
                    node=node,
                    score=result.score
                )
                nodes_with_scores.append(node_with_score)
        
        print(f"🔍 Retrieved {len(nodes_with_scores)} relevant documents")
        return nodes_with_scores


def create_custom_retriever(
    client: QdrantClient,
    collection_name: str = "medmax_pubmed",
    top_k: int = 5,
    score_threshold: float = 0.7
) -> QdrantRetriever:
    """Factory function to create custom Qdrant retriever."""
    retriever = QdrantRetriever(
        client=client,
        collection_name=collection_name,
        top_k=top_k,
        score_threshold=score_threshold
    )
    
    print(f"🎯 Custom retriever created (top_k={top_k}, threshold={score_threshold})")
    return retriever
