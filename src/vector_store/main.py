"""Main script for MedMax vector store operations."""

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .client import setup_qdrant_client, collection_exists, collection_has_data, create_collection
from .loader import load_pubmed_data, load_pubmedqa_parquet_data, format_pubmed_for_embedding
from .embed import generate_openai_embeddings, generate_embeddings
from .ingestion import upload_pubmed_to_qdrant


def populate_qdrant(
    collection_name: str = "medmax_pubmed",
    data_source: str = None,
    limit: int = None,
    force_reindex: bool = False,
    use_parquet: bool = False,
    use_huggingface_embeddings: bool = False,
    embedding_model: str = None,
    embedding_device: str = "auto",
    embedding_batch_size: int = 32
):
    """
    Populate Qdrant with PubMed data using OpenAI or HuggingFace embeddings.
    
    Args:
        collection_name: Name of Qdrant collection
        data_source: Path to data file or directory
        limit: Limit number of records to process
        force_reindex: Force recreation of collection
        use_parquet: Use parquet files instead of JSONL
        use_huggingface_embeddings: Use HuggingFace embeddings instead of OpenAI
        embedding_model: Model name (auto-detected if None)
        embedding_device: Device for HuggingFace models ("cuda", "cpu", "auto")
        embedding_batch_size: Batch size for embedding generation
    """
    print("=== MedMax Vector Store Creation ===")
    print(f"Collection name: {collection_name}")

    if use_huggingface_embeddings:
        if embedding_model is None:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Embedding: HuggingFace {embedding_model} (FREE, Local)")
        print(f"Device: {embedding_device}")
        print(f"Batch size: {embedding_batch_size}")
    else:
        if embedding_model is None:
            embedding_model = "text-embedding-3-small"
        print(f"Embedding: OpenAI {embedding_model} (Paid, API)")

    if use_parquet:
        print("Data source: PubMedQA parquet files (unlabeled + labeled + artificial)")
        data_source = "data"  # Directory containing parquet files
    else:
        print(f"Data source: {data_source}")
        
    if limit:
        print(f"TEST MODE: Limited to {limit} records" + (" per dataset" if use_parquet else ""))
    if force_reindex:
        print("FORCE REINDEX: Will recreate collection even if data exists")
    
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
    
    # Load data based on source type
    print("\n2. Loading PubMed data...")
    if use_parquet:
        pubmed_records, stats = load_pubmedqa_parquet_data(
            data_dir=data_source,
            limit_per_dataset=limit
        )
        if "error" in stats:
            print("Failed to load parquet data. Exiting.")
            return False
    else:
        # Check if data file exists
        if not os.path.exists(data_source):
            print(f"Error: Data file not found at {data_source}")
            return False
        pubmed_records = load_pubmed_data(data_source, limit=limit)
    
    if not pubmed_records:
        print("No valid records found. Exiting.")
        return False
    
    # Format data for embedding
    print("\n3. Formatting data for embedding...")
    texts = format_pubmed_for_embedding(pubmed_records)
    
    # Generate embeddings
    print("\n4. Generating embeddings...")
    print(f"   Using: {'HuggingFace (Local)' if use_huggingface_embeddings else 'OpenAI (API)'}")
    try:
        embeddings = generate_embeddings(
            texts=texts,
            model=embedding_model,
            use_huggingface=use_huggingface_embeddings,
            batch_size=embedding_batch_size,
            device=embedding_device
        )
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        if use_huggingface_embeddings:
            print("\nTroubleshooting:")
            print("  1. Install: pip install sentence-transformers torch")
            print("  2. Check GPU: python -c 'import torch; print(torch.cuda.is_available())'")
            print("  3. Try CPU: --embedding-device cpu")
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
    print(f"Embedding model: {embedding_model}")

    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MedMax Vector Store - Populate Qdrant with PubMed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI embeddings (default, paid)
  python -m src.vector_store.main populate --limit 10

  # HuggingFace embeddings (free, local)
  python -m src.vector_store.main populate --use-huggingface-embeddings --limit 10
  
  # HuggingFace with custom model
  python -m src.vector_store.main populate --use-huggingface-embeddings --embedding-model all-MiniLM-L6-v2 --limit 100
  
  # HuggingFace with GPU
  python -m src.vector_store.main populate --use-huggingface-embeddings --embedding-device cuda --limit 1000
  
  # Full dataset with HuggingFace
  python -m src.vector_store.main populate --use-parquet --use-huggingface-embeddings --embedding-model all-MiniLM-L6-v2
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
        help="Path to PubMed JSONL data file (ignored if --use-parquet is specified)"
    )
    
    parser.add_argument(
        "--use-parquet",
        action="store_true",
        help="Use PubMedQA parquet files instead of JSONL (loads from data/ directory)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (per dataset if using parquet)"
    )
    
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing even if collection already has data"
    )

    parser.add_argument(
        "--use-huggingface-embeddings",
        action="store_true",
        help="Use HuggingFace embeddings instead of OpenAI (FREE, local processing)"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="""Embedding model to use:
        - OpenAI: text-embedding-3-small (default), text-embedding-3-large
        - HuggingFace: all-MiniLM-L6-v2 (default), bge-base-en-v1.5, bge-large-en-v1.5
        """
    )
    
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for HuggingFace embeddings (default: auto - detects GPU automatically)"
    )
    
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for HuggingFace embedding generation (default: 32)"
    )
    
    
    args = parser.parse_args(argv)
    
    # Determine data source
    if args.use_parquet:
        # Check if parquet files exist
        data_dir = Path("data")
        required_files = [
            data_dir / "pqa_unlabeled_train.parquet",
            data_dir / "pqa_labeled_train.parquet", 
            data_dir / "pqa_artificial_train.parquet"
        ]
        missing_files = [str(f) for f in required_files if not f.exists()]
        if missing_files:
            print(f"Error: Missing parquet files: {missing_files}")
            return 1
        data_source = str(data_dir)
    else:
        # Check if JSONL file exists
        if not Path(args.data_path).exists():
            print(f"Error: Data file not found at {args.data_path}")
            return 1
        data_source = args.data_path
    
    if args.operation == "populate":
        success = populate_qdrant(
            collection_name=args.collection_name,
            data_source=data_source,
            limit=args.limit,
            force_reindex=args.force_reindex,
            use_parquet=args.use_parquet,
            use_huggingface_embeddings=args.use_huggingface_embeddings,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size
        )
        if not success:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
