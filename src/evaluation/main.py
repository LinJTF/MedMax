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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize evaluator
    evaluator = MedREQALEvaluator(
        collection_name=args.collection_name,
        engine_type=args.engine_type,
        delay_between_queries=args.delay
    )
    
    print("🚀 Starting MedREQAL evaluation...")
    print(f"📁 Dataset: {args.csv_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🗃️ Collection: {args.collection_name}")
    if args.limit:
        print(f"🔢 Limit: {args.limit} questions")
    
    # Run evaluation
    try:
        _, metrics = evaluator.evaluate_dataset(
            csv_path=args.csv_path,
            output_path=output_dir / f"rag_results_{timestamp}.csv",
            limit=args.limit
        )
        
        # Print summary
        evaluator.print_summary(metrics)
        
        # Save metrics
        metrics_file = output_dir / f"rag_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n💾 Detailed metrics saved to {metrics_file}")
        
        # Compare with baseline if provided
        if args.baseline_accuracy:
            print(f"\n🔍 Comparing with baseline accuracy: {args.baseline_accuracy:.3f}")
            
            if 'overall' in metrics:
                rag_accuracy = metrics['overall']['accuracy']
                comparison = compare_with_baseline(rag_accuracy, args.baseline_accuracy)
                
                print("📊 Comparison Results:")
                print(f"   RAG Accuracy: {rag_accuracy:.3f}")
                print(f"   Baseline Accuracy: {args.baseline_accuracy:.3f}")
                print(f"   Improvement: {comparison['improvement']:.3f}")
                print(f"   Relative Improvement: {comparison['relative_improvement']:.1f}%")
                
                if comparison['improvement'] > 0:
                    print("✅ RAG system outperforms baseline!")
                elif comparison['improvement'] < 0:
                    print("❌ RAG system underperforms baseline")
                else:
                    print("➡️ RAG system matches baseline performance")
                
                # Save comparison
                comparison_file = output_dir / f"baseline_comparison_{timestamp}.json"
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2)
                print(f"💾 Comparison saved to {comparison_file}")
        
        print("\n✅ Evaluation completed successfully!")
        print(f"📊 Results available in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
