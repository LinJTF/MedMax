"""Metrics calculation for medical Q&A evaluation."""

from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np


def normalize_verdict(verdict: str) -> str:
    """Normalize verdict labels to standard format."""
    verdict = str(verdict).upper().strip()
    
    # Handle different verdict formats
    if 'ENOUGH' in verdict or 'YES' in verdict or 'BENEFIT' in verdict:
        return 'YES'
    elif 'NOT ENOUGH' in verdict or 'NO' in verdict or 'HARM' in verdict:
        return 'NO'
    elif 'UNCERTAIN' in verdict or 'UNCLEAR' in verdict:
        return 'UNCERTAIN'
    else:
        return verdict


def extract_verdict_from_response(response: str) -> str:
    """Extract verdict from RAG response text."""
    response = response.upper().strip()
    
    # Look for explicit verdicts
    if 'YES' in response and ('BENEFIT' in response or 'EFFECTIVE' in response):
        return 'YES'
    elif 'NO' in response and ('HARM' in response or 'NOT EFFECTIVE' in response):
        return 'NO'
    elif 'NOT ENOUGH' in response or 'INSUFFICIENT' in response:
        return 'NOT ENOUGH INFORMATION'
    elif 'UNCERTAIN' in response or 'UNCLEAR' in response:
        return 'UNCERTAIN'
    else:
        # Default fallback based on keywords
        if any(word in response for word in ['EFFECTIVE', 'BENEFICIAL', 'IMPROVES']):
            return 'YES'
        elif any(word in response for word in ['INEFFECTIVE', 'HARMFUL', 'WORSE']):
            return 'NO'
        else:
            return 'NOT ENOUGH INFORMATION'


def calculate_accuracy(true_labels: List[str], predicted_labels: List[str]) -> float:
    """Calculate accuracy score."""
    # Normalize labels
    true_normalized = [normalize_verdict(label) for label in true_labels]
    pred_normalized = [normalize_verdict(label) for label in predicted_labels]
    
    return accuracy_score(true_normalized, pred_normalized)


def calculate_f1_score(true_labels: List[str], predicted_labels: List[str], average: str = 'weighted') -> float:
    """Calculate F1 score."""
    # Normalize labels
    true_normalized = [normalize_verdict(label) for label in true_labels]
    pred_normalized = [normalize_verdict(label) for label in predicted_labels]
    
    return f1_score(true_normalized, pred_normalized, average=average)


def evaluate_predictions(
    true_labels: List[str], 
    predicted_labels: List[str], 
    detailed: bool = True
) -> Dict[str, Any]:
    """Comprehensive evaluation of predictions."""
    
    # Normalize labels
    true_normalized = [normalize_verdict(label) for label in true_labels]
    pred_normalized = [normalize_verdict(label) for label in predicted_labels]
    
    # Basic metrics
    accuracy = accuracy_score(true_normalized, pred_normalized)
    f1_weighted = f1_score(true_normalized, pred_normalized, average='weighted')
    f1_macro = f1_score(true_normalized, pred_normalized, average='macro')
    
    results = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'total_samples': len(true_labels)
    }
    
    if detailed:
        # Classification report
        report = classification_report(
            true_normalized, 
            pred_normalized, 
            output_dict=True
        )
        results['classification_report'] = report
        
        # Per-class analysis
        unique_labels = list(set(true_normalized + pred_normalized))
        results['per_class'] = {}
        
        for label in unique_labels:
            true_count = true_normalized.count(label)
            pred_count = pred_normalized.count(label)
            results['per_class'][label] = {
                'true_count': true_count,
                'predicted_count': pred_count
            }
    
    return results


def calculate_category_performance(
    df: pd.DataFrame, 
    true_col: str = 'verdicts', 
    pred_col: str = 'rag_predictions',
    category_col: str = 'category'
) -> Dict[str, Dict[str, float]]:
    """Calculate performance metrics by medical category."""
    
    results = {}
    
    for category in df[category_col].unique():
        category_df = df[df[category_col] == category]
        
        if len(category_df) > 0:
            true_labels = category_df[true_col].tolist()
            pred_labels = category_df[pred_col].tolist()
            
            results[category] = {
                'accuracy': calculate_accuracy(true_labels, pred_labels),
                'f1_weighted': calculate_f1_score(true_labels, pred_labels, 'weighted'),
                'f1_macro': calculate_f1_score(true_labels, pred_labels, 'macro'),
                'sample_count': len(category_df)
            }
    
    return results


def create_confusion_matrix_data(true_labels: List[str], predicted_labels: List[str]) -> pd.DataFrame:
    """Create confusion matrix as DataFrame."""
    from sklearn.metrics import confusion_matrix
    
    # Normalize labels
    true_normalized = [normalize_verdict(label) for label in true_labels]
    pred_normalized = [normalize_verdict(label) for label in predicted_labels]
    
    # Get unique labels
    unique_labels = sorted(list(set(true_normalized + pred_normalized)))
    
    # Create confusion matrix
    cm = confusion_matrix(true_normalized, pred_normalized, labels=unique_labels)
    
    # Convert to DataFrame
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    
    return cm_df


def compare_with_baseline(rag_accuracy: float, baseline_accuracy: float) -> Dict[str, float]:
    """Compare RAG performance with baseline performance."""
    improvement = rag_accuracy - baseline_accuracy
    relative_improvement = (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
    
    return {
        'rag_accuracy': rag_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement': improvement,
        'relative_improvement': relative_improvement
    }
