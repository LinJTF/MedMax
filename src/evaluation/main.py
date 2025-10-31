"""
Main evaluation script for RAG vs Zero-shot performance evaluation.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import AFTER path setup to avoid circular imports
from src.evaluation.medreqal_evaluator import MedREQALEvaluator
from src.evaluation.metrics import compare_with_baseline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system on MedREQAL or PubMedQA dataset"
    )
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, choices=["medreqal", "pubmedqa"], required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    
    # Qdrant arguments
    parser.add_argument("--collection_name", type=str, default="medmax_pubmed_full")
    
    # Evaluation arguments
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--engine_type", type=str, choices=["simple", "standard", "enhanced"], default="standard")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--mode", type=str, choices=["rag", "zero_shot"], default="rag")
    
    # Model arguments
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--use_ollama", action="store_true")
    parser.add_argument("--use_huggingface", action="store_true")
    
    # Comparison arguments
    parser.add_argument("--baseline_accuracy", type=float, default=None)
    
    return parser.parse_args()


def run_evaluation(args):
    """Run the evaluation with given arguments."""
    print(f"\n{'='*70}")
    print(f"STARTING EVALUATION")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.llm_model}")
    print(f"Ollama: {args.use_ollama}")
    print(f"HuggingFace: {args.use_huggingface}")
    if args.limit:
        print(f"Limit: {args.limit} questions")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    model_name_clean = args.llm_model.replace(":", "_").replace("-", "_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_prefix = f"{model_name_clean}_{args.dataset_type}_{args.mode}"

    # Initialize evaluator
    evaluator = MedREQALEvaluator(
        collection_name=args.collection_name,
        engine_type=args.engine_type,
        delay_between_queries=args.delay,
        mode=args.mode,
        llm_model=args.llm_model,
        dataset_type=args.dataset_type,
        use_ollama=args.use_ollama,
        use_huggingface=args.use_huggingface
    )

    # Run evaluation
    try:
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

        # Save metrics
        metrics_file = output_dir / f"{result_prefix}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n Metrics saved to {metrics_file}")

        # Compare with baseline if provided
        if args.baseline_accuracy and 'overall' in metrics:
            eval_accuracy = metrics['overall']['accuracy']
            comparison = compare_with_baseline(eval_accuracy, args.baseline_accuracy)
            
            print(f"\nBaseline Comparison:")
            print(f"  Evaluated: {eval_accuracy:.3f}")
            print(f"  Baseline: {args.baseline_accuracy:.3f}")
            print(f"  Improvement: {comparison['improvement']:.3f} ({comparison['relative_improvement']:.1f}%)")
            
            comparison_file = output_dir / f"baseline_comparison_{timestamp}.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)

        print(f"\n Evaluation completed successfully!")
        print(f"Results: {output_dir}\n")
        return 0

    except Exception as e:
        print(f"\n Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    args = parse_arguments()
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
