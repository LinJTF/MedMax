"""Main CLI for RAG system queries."""

from langfuse import observe
import argparse
import sys
from typing import Optional, Sequence

from ..observability.langfuse_config import update_trace_metadata, update_span_metadata
from .client import setup_rag_client, setup_qdrant_client
from .models import configure_global_settings, setup_llm, setup_embedding_model
from .query_engine import create_simple_query_engine, create_query_engine, create_standard_query_engine, create_enhanced_query_engine, enhanced_query_engine
from .retriever import create_custom_retriever


def patch_query_engine_with_tracing(query_engine):
    """Add tracing to existing query engine."""
    original_query = query_engine.query
    
    @observe()
    def traced_query(question):
        return original_query(question)
    
    query_engine.query = traced_query
    return query_engine

@observe()
def interactive_query_session_rag(query_engine, collection_name: str):
    """Run an interactive query session."""
    update_span_metadata({
        "operation": "interactive_rag_session",
        "collection_name": collection_name,
        "session_type": "interactive",
        "mode": "rag"
    })
    print(f"\nMedMax RAG Interactive Session")
    print(f"Collection: {collection_name}")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 50)
    
    while True:
        try:
            # Get user query
            query = input("\nEnter your medical question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Ending session. Goodbye!")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            print("\nProcessing your query...")
            
            # Query the system
            response = query_engine.query(query)
            
            # Display response
            print("\n" + "=" * 50)
            print("ANSWER:")
            print("-" * 20)
            print(response.response)
            
            # Show sources if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\nSOURCES ({len(response.source_nodes)} found):")
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
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again with a different question.")
            
@observe()
def interactive_query_session_zero_shot(llm_model: str):
    """Interactive session for zero-shot mode."""
    update_span_metadata({
        "operation": "interactive_zero_shot_session",
        "llm_model": llm_model,
        "session_type": "interactive",
        "mode": "zero_shot"
    })
    print(f"\nMedMax Zero-Shot Interactive Session")
    print(f"Model: {llm_model}")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 50)
    llm = setup_llm(model=llm_model)
    while True:
        try:
            query = input("\nEnter your medical question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Ending session. Goodbye!")
                break
            if not query:
                print("Please enter a valid question.")
                continue
            print("\nProcessing your query...")
            prompt = (
                f"Medical Question: {query}\n"
                "Please answer and provide a clear verdict: SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION."
            )
            response = llm.complete(prompt)
            answer = response.text if hasattr(response, "text") else str(response)
            print("\n" + "=" * 50)
            print("ANSWER:")
            print("-" * 20)
            print(answer)
            print("=" * 50)
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again with a different question.")

@observe()
def single_query_rag(query_engine, question: str, verbose: bool = False):
    """Process a single query and return results."""
    update_span_metadata({
        "operation": "single_rag_query",
        "verbose_mode": verbose,
        "query_type": "single_query",
        "mode": "rag"
    })
    print(f"\nQuery: {question}")
    print("Processing...")
    
    try:
        response = query_engine.query(question)
        
        print("\n" + "=" * 50)
        print("ANSWER:")
        print("-" * 20)
        print(response.response)
        
        if verbose and hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\nSOURCES ({len(response.source_nodes)} found):")
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
        print(f"Error: {e}")
        return False

@observe()
def single_query_zero_shot(question: str, llm_model: str):
    """Single query for zero-shot mode."""
    update_span_metadata({
        "operation": "single_zero_shot_query",
        "llm_model": llm_model,
        "query_type": "single_query",
        "mode": "zero_shot"
    })
    print(f"\nQuery: {question}")
    print("Processing...")
    try:
        llm = setup_llm(model=llm_model)
        prompt = (
            f"Medical Question: {question}\n"
            "Please answer and provide a clear verdict: SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION."
        )
        response = llm.complete(prompt)
        answer = response.text if hasattr(response, "text") else str(response)
        print("\n" + "=" * 50)
        print("ANSWER:")
        print("-" * 20)
        print(answer)
        print("=" * 50)
        return True
    except Exception as e:
        print(f"Error: {e}")
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
  python -m src.rag.main interactive --mode rag # Interactive session with RAG system
  python -m src.rag.main interactive --mode zero_shot --model gpt-4o-mini # Interactive session with zero shot system and specific model
  python -m src.rag.main query "What is diabetes?" --mode rag                      # Single RAG query
  python -m src.rag.main query "Treatment for hypertension" --mode zero_shot       # Single zero-shot query
        """
    )
    
    parser.add_argument(
        "cli_mode",
        choices=["interactive", "query"],
        help="CLI mode: 'interactive' for session, 'query' for single question"
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (required for 'query' mode)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["rag", "zero_shot"],
        default="rag",
        help="Query mode: 'rag' (retrieval-augmented) or 'zero_shot' (LLM only)"
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
    

    if args.cli_mode == "query" and not args.question:
        print("Error: Question is required for 'query' mode")
        return 1
    
    try:
        print(f"Starting MedMax system in {args.mode.upper()} mode...")

        if args.mode == "rag":
            # Configure global settings
            configure_global_settings()
            # Setup RAG client
            vector_store, index = setup_rag_client(args.collection_name)
            # Create query engine based on type
            if args.engine_type == "simple":
                query_engine = create_simple_query_engine(index, top_k=args.top_k)
            elif args.engine_type == "enhanced":
                query_engine = create_enhanced_query_engine(
                    index, 
                    collection_name=args.collection_name,
                    top_k=args.top_k, 
                    llm_model=args.model,
                    verbose=args.verbose
                )
            else:  # standard
                query_engine = create_standard_query_engine(
                    index,
                    collection_name=args.collection_name,
                    top_k=args.top_k,
                    llm_model=args.model
                )
            query_engine = patch_query_engine_with_tracing(query_engine)
            print("RAG system ready!")

            # Run based on CLI mode
            if args.cli_mode == "interactive":
                interactive_query_session_rag(query_engine, args.collection_name)
            else:
                success = single_query_rag(query_engine, args.question, args.verbose)
                return 0 if success else 1

        else:  # zero_shot
            print("Zero-Shot mode ready!")
            if args.cli_mode == "interactive":
                interactive_query_session_zero_shot(args.model)
            else:
                success = single_query_zero_shot(args.question, args.model)
                return 0 if success else 1

    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
