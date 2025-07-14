"""Main script for MedMax vector store operations."""

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .client import setup_qdrant_client, collection_exists, collection_has_data, create_collection
from .loader import load_pubmed_data, format_pubmed_for_embedding
from .embed import generate_openai_embeddings
from .ingestion import upload_pubmed_to_qdrant


def populate_qdrant(
    collection_name: str = "medmax_pubmed",
    jsonl_path: str = "data/PubMed-compact/pubmedqa.jsonl",
    limit: int = None,
    force_reindex: bool = False
):
    """Populate Qdrant with PubMed data."""
    print("=== MedMax Vector Store Creation ===")
    print(f"Collection name: {collection_name}")
    print(f"Data source: {jsonl_path}")
    if limit:
        print(f"TEST MODE: Limited to {limit} records")
    if force_reindex:
        print(f"FORCE REINDEX: Will recreate collection even if data exists")
    
    # Check if data file exists
    if not os.path.exists(jsonl_path):
        print(f"Error: Data file not found at {jsonl_path}")
        return False
    
    # Setup Qdrant client
    print("\n1. Setting up Qdrant client...")
    try:
        client = setup_qdrant_client()
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return False
    
    # Check if collection already has data (unless force reindex)
    if not force_reindex and collection_has_data(client, collection_name):
        print(f"Collection '{collection_name}' already has data!")
        print("Use --force-reindex flag to recreate the collection.")
        return True
    
    # Load PubMed data
    print("\n2. Loading PubMed data...")
    pubmed_records = load_pubmed_data(jsonl_path, limit=limit)
    
    if not pubmed_records:
        print("No valid records found. Exiting.")
        return False
    
    # Format data for embedding
    print("\n3. Formatting data for embedding...")
    texts = format_pubmed_for_embedding(pubmed_records)
    
    # Generate embeddings
    print("\n4. Generating embeddings...")
    try:
        embeddings = generate_openai_embeddings(texts)
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        return False
    
    # Create or recreate collection
    print("\n5. Setting up Qdrant collection...")
    vector_size = len(embeddings[0]) if embeddings else 1536
    
    if force_reindex and collection_exists(client, collection_name):
        print(f"Recreating collection '{collection_name}' due to force reindex...")
        client.delete_collection(collection_name)
    
    if not collection_exists(client, collection_name):
        create_collection(client, collection_name, vector_size)
    else:
        print(f"Using existing collection '{collection_name}'")
    
    # Upload to Qdrant
    print("\n6. Uploading data to Qdrant...")
    try:
        upload_pubmed_to_qdrant(client, collection_name, pubmed_records, embeddings)
    except Exception as e:
        print(f"Failed to upload data: {e}")
        return False
    
    print(f"\nSuccessfully created vector store with {len(pubmed_records)} records!")
    print(f"Collection: {collection_name}")
    print(f"Vector size: {vector_size}")
    
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MedMax Vector Store - Populate Qdrant with PubMed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.vector_store.main populate                                    # Full dataset
  python -m src.vector_store.main populate --limit 10                         # Test with 10 records
  python -m src.vector_store.main populate --collection-name my_collection    # Custom collection
  python -m src.vector_store.main populate --force-reindex                    # Force recreate even if exists
        """
    )
    
    parser.add_argument(
        "operation",
        choices=["populate"],
        help="Operation to perform (currently only 'populate' is supported)"
    )
    
    parser.add_argument(
        "--collection-name",
        default="medmax_pubmed",
        help="Name of the Qdrant collection to create/use (default: medmax_pubmed)"
    )
    
    parser.add_argument(
        "--data-path",
        default="data/PubMed-compact/pubmedqa.jsonl",
        help="Path to PubMed JSONL data file"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (useful for testing, e.g., --limit 10)"
    )
    
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing even if collection already has data"
    )
    
    args = parser.parse_args(argv)
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found at {args.data_path}")
        return 1
    
    if args.operation == "populate":
        success = populate_qdrant(
            args.collection_name, 
            args.data_path, 
            args.limit,
            args.force_reindex
        )
        if not success:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
