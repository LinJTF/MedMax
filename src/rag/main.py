"""Main CLI for RAG system queries."""

import argparse
import sys
from typing import Optional, Sequence

from .client import setup_rag_client, setup_qdrant_client
from .models import configure_global_settings, setup_llm, setup_embedding_model
from .query_engine import create_simple_query_engine, create_query_engine, enhanced_query_engine
from .retriever import create_custom_retriever


def interactive_query_session(query_engine, collection_name: str):
    """Run an interactive query session."""
    print(f"\n🎯 MedMax RAG Interactive Session")
    print(f"📦 Collection: {collection_name}")
    print("💡 Type 'quit' or 'exit' to end the session")
    print("=" * 50)
    
    while True:
        try:
            # Get user query
            query = input("\n🔍 Enter your medical question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Ending session. Goodbye!")
                break
            
            if not query:
                print("❌ Please enter a valid question.")
                continue
            
            print("\n🤖 Processing your query...")
            
            # Query the system
            response = query_engine.query(query)
            
            # Display response
            print("\n" + "=" * 50)
            print("📋 ANSWER:")
            print("-" * 20)
            print(response.response)
            
            # Show sources if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n📚 SOURCES ({len(response.source_nodes)} found):")
                print("-" * 20)
                for i, node in enumerate(response.source_nodes, 1):
                    score = getattr(node, 'score', 'N/A')
                    metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                    question = metadata.get('question', 'N/A')
                    decision = metadata.get('final_decision', 'N/A')
                    record_id = metadata.get('record_id', 'N/A')
                    print(f"{i}. Score: {score:.3f} | ID: {record_id} | Decision: {decision}")
                    print(f"   Question: {question[:100]}...")
            
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("Please try again with a different question.")


def single_query(query_engine, question: str, verbose: bool = False):
    """Process a single query and return results."""
    print(f"\n🔍 Query: {question}")
    print("🤖 Processing...")
    
    try:
        response = query_engine.query(question)
        
        print("\n" + "=" * 50)
        print("📋 ANSWER:")
        print("-" * 20)
        print(response.response)
        
        if verbose and hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\n📚 SOURCES ({len(response.source_nodes)} found):")
            print("-" * 20)
            for i, node in enumerate(response.source_nodes, 1):
                score = getattr(node, 'score', 'N/A')
                metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
                question = metadata.get('question', 'N/A')
                decision = metadata.get('final_decision', 'N/A')
                record_id = metadata.get('record_id', 'N/A')
                contexts = metadata.get('contexts', [])
                long_answer = metadata.get('long_answer', 'N/A')
                print(f"{i}. Score: {score:.3f} | Record ID: {record_id}")
                print(f"   Question: {question}")
                print(f"   Decision: {decision}")
                print(f"   Contexts: {len(contexts)} available")
                print(f"   Long Answer: {long_answer[:100]}...")
                print()
        
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main function for RAG CLI."""
    parser = argparse.ArgumentParser(
        description="MedMax RAG System - Query medical knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.rag.main interactive                                    # Interactive session
  python -m src.rag.main query "What is diabetes?"                      # Single query
  python -m src.rag.main query "Treatment for hypertension" --verbose   # Verbose output
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["interactive", "query"],
        help="Mode of operation: 'interactive' for session, 'query' for single question"
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (required for 'query' mode)"
    )
    
    parser.add_argument(
        "--collection-name",
        default="medmax_pubmed",
        help="Qdrant collection name to query (default: medmax_pubmed)"
    )
    
    parser.add_argument(
        "--engine-type",
        choices=["simple", "standard", "enhanced"],
        default="standard",
        help="Type of query engine to use (default: standard)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed source information"
    )
    
    args = parser.parse_args(argv)
    
    # Validate arguments
    if args.mode == "query" and not args.question:
        print("❌ Error: Question is required for 'query' mode")
        return 1
    
    try:
        print("🚀 Starting MedMax RAG system...")
        
        # Configure global settings
        configure_global_settings()
        
        # Setup RAG client
        vector_store, index = setup_rag_client(args.collection_name)
        
        # Create query engine based on type
        if args.engine_type == "simple":
            query_engine = create_simple_query_engine(index, top_k=args.top_k)
        elif args.engine_type == "enhanced":
            query_engine = enhanced_query_engine(index, top_k=args.top_k, verbose=args.verbose)
        else:  # standard
            # Setup custom retriever for standard mode
            qdrant_client = setup_qdrant_client()
            custom_retriever = create_custom_retriever(
                qdrant_client, 
                args.collection_name, 
                top_k=args.top_k
            )
            query_engine = create_query_engine(
                index, 
                retriever=custom_retriever, 
                llm_model=args.model
            )
        
        print("✅ RAG system ready!")
        
        # Run based on mode
        if args.mode == "interactive":
            interactive_query_session(query_engine, args.collection_name)
        else:  # query mode
            success = single_query(query_engine, args.question, args.verbose)
            return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
