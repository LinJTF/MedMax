"""
Main evaluation script for comparing RAG vs Zero-shot performance on MedREQAL dataset.

This script evaluates the RAG system against the MedREQAL benchmark and compares
the results with zero-shot performance reported in academic literature.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langfuse import observe
from src.evaluation.medreqal_evaluator import MedREQALEvaluator
from src.evaluation.metrics import compare_with_baseline


@observe(name="Evaluation flow")
def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system on MedREQAL or PubMedQA dataset")
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to dataset file (CSV for MedREQAL, parquet for PubMedQA)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["medreqal", "pubmedqa"],
        required=True,
        help="Type of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--collection_name", 
        type=str, 
        default="medmax_pubmed_full",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of questions to evaluate (for testing)"
    )
    parser.add_argument(
        "--engine_type", 
        type=str, 
        choices=["simple", "standard", "enhanced"],
        default="standard",
        help="Type of query engine to use (default: standard)"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Delay between queries in seconds"
    )
    parser.add_argument(
        "--baseline_accuracy", 
        type=float, 
        default=None,
        help="Baseline accuracy for comparison (e.g., zero-shot performance)"
    )
    
    parser.add_argument(
    "--mode",
    type=str,
    choices=["rag", "zero_shot"],
    default="rag",
    help="Evaluation mode: rag or zero_shot"
    )
    
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (e.g., gpt-4o-mini, mistral:7b)"
    )
    
    parser.add_argument(
        "--use_ollama",
        action="store_true",
        help="Use Ollama for local models instead of OpenAI"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output file prefix based on mode and dataset
    result_prefix = f"{args.dataset_type}_{args.mode}"

    # Initialize evaluator with dataset type
    evaluator = MedREQALEvaluator(
        collection_name=args.collection_name,
        engine_type=args.engine_type,
        delay_between_queries=args.delay,
        mode=args.mode,
        llm_model=args.llm_model,
        dataset_type=args.dataset_type,
        use_ollama=args.use_ollama
    )

    print(f"Starting {args.dataset_type.upper()} evaluation in {args.mode.upper()} mode...")
    print(f"Dataset: {args.data_path}")
    print(f"Output: {output_dir}")
    print(f"Collection: {args.collection_name}")
    if args.limit:
        print(f"Limit: {args.limit} questions")

    try:
        # Route to appropriate evaluation method based on dataset type
        if args.dataset_type == "pubmedqa":
            _, metrics = evaluator.evaluate_pubmedqa_dataset(
                parquet_path=args.data_path,
                output_path=output_dir / f"{result_prefix}_results_{timestamp}.csv",
                limit=args.limit
            )
            evaluator.print_pubmedqa_summary(metrics)
        else:  # medreqal
            _, metrics = evaluator.evaluate_dataset(
                csv_path=args.data_path,
                output_path=output_dir / f"{result_prefix}_results_{timestamp}.csv",
                limit=args.limit
            )
            evaluator.print_summary(metrics)

        metrics_file = output_dir / f"{result_prefix}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nDetailed metrics saved to {metrics_file}")

        if args.baseline_accuracy:
            print(f"\nComparing with baseline accuracy: {args.baseline_accuracy:.3f}")
            if 'overall' in metrics:
                eval_accuracy = metrics['overall']['accuracy']
                comparison = compare_with_baseline(eval_accuracy, args.baseline_accuracy)
                print("Comparison Results:")
                print(f"   Evaluated Accuracy: {eval_accuracy:.3f}")
                print(f"   Baseline Accuracy: {args.baseline_accuracy:.3f}")
                print(f"   Improvement: {comparison['improvement']:.3f}")
                print(f"   Relative Improvement: {comparison['relative_improvement']:.1f}%")
                if comparison['improvement'] > 0:
                    print(f"{args.mode.upper()} system outperforms baseline!")
                elif comparison['improvement'] < 0:
                    print(f"{args.mode.upper()} system underperforms baseline")
                else:
                    print(f"{args.mode.upper()} system matches baseline performance")
                comparison_file = output_dir / f"baseline_comparison_{timestamp}.json"
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2)
                print(f"Comparison saved to {comparison_file}")

        print("\nEvaluation completed successfully!")
        print(f"Results available in: {output_dir}")

    except Exception as e:
        print(f" Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
