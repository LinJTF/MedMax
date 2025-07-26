"""
Main evaluation script for comparing RAG vs Zero-shot performance on MedREQAL dataset.

This script evaluates the RAG system against the MedREQAL benchmark and compares
the results with zero-shot performance reported in academic literature.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

from .medreqal_evaluator import MedREQALEvaluator
from .metrics import compare_with_baseline


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system on MedREQAL dataset")
    parser.add_argument(
        "--csv_path", 
        type=str, 
        required=True,
        help="Path to MedREQAL CSV file"
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
        default="medmax_pubmed",
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
        help="LLM model name for zero-shot mode"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output file prefix based on mode
    result_prefix = "rag" if args.mode == "rag" else "zero_shot"

    # Initialize evaluator with mode and llm_model
    evaluator = MedREQALEvaluator(
        collection_name=args.collection_name,
        engine_type=args.engine_type,
        delay_between_queries=args.delay,
        mode=args.mode,
        llm_model=args.llm_model
    )

    print(f"Starting MedREQAL evaluation in {args.mode.upper()} mode...")
    print(f"Dataset: {args.csv_path}")
    print(f"Output: {output_dir}")
    print(f"Collection: {args.collection_name}")
    if args.limit:
        print(f"Limit: {args.limit} questions")

    try:
        _, metrics = evaluator.evaluate_dataset(
            csv_path=args.csv_path,
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
