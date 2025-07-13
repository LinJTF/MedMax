"""Evaluation module for comparing RAG performance with MedREQAL benchmarks."""

from .medreqal_evaluator import MedREQALEvaluator
from .metrics import (
    calculate_accuracy, 
    calculate_f1_score, 
    evaluate_predictions,
    extract_verdict_from_response,
    calculate_category_performance,
    compare_with_baseline
)

__all__ = [
    "MedREQALEvaluator",
    "calculate_accuracy", 
    "calculate_f1_score",
    "evaluate_predictions",
    "extract_verdict_from_response",
    "calculate_category_performance",
    "compare_with_baseline"
]
