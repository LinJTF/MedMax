"""MedREQAL evaluator for comparing RAG performance with zero-shot results."""

from langfuse import observe
from src.observability.langfuse_config import create_session

import pandas as pd
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

# RAG system imports
from ..rag.client import setup_rag_client
from ..rag.models import configure_global_settings
from ..rag.query_engine import create_simple_query_engine, create_standard_query_engine, create_enhanced_query_engine
from .metrics import evaluate_predictions, extract_verdict_from_response, calculate_category_performance


class MedREQALEvaluator:
    """Evaluator for MedREQAL dataset using RAG system."""
    
    def __init__(
        self,
        collection_name: str = "medmax_pubmed",
        engine_type: str = "standard",  # Changed default to standard
        delay_between_queries: float = 1.0
    ):
        """Initialize MedREQAL evaluator."""
        self.collection_name = collection_name
        self.engine_type = engine_type
        self.delay_between_queries = delay_between_queries
        self.query_engine = None
        self.results = []
        
    def setup_rag_system(self):
        """Setup RAG system for evaluation."""
        print("Setting up RAG system for MedREQAL evaluation...")
        
        # Configure global settings
        configure_global_settings()
        
        # Setup RAG client
        _, index = setup_rag_client(self.collection_name)
        
        # Create query engine based on type
        if self.engine_type == "simple":
            self.query_engine = create_simple_query_engine(index, top_k=5)
        elif self.engine_type == "enhanced":
            self.query_engine = create_enhanced_query_engine(
                index, 
                collection_name=self.collection_name,
                top_k=5
            )
        else:  # standard (default)
            self.query_engine = create_standard_query_engine(
                index,
                collection_name=self.collection_name,
                top_k=5
            )
        
        print("RAG system ready for evaluation")
    
    def load_medreqal_data(self, csv_path: str) -> pd.DataFrame:
        """Load MedREQAL dataset from CSV."""
        print(f"Loading MedREQAL data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} questions from MedREQAL dataset")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def format_question_with_context(self, row: pd.Series) -> str:
        """Format question with background context for RAG query."""
        question = row['question']
        background = row.get('background', '')
        objective = row.get('objective', '')
        
        # Create comprehensive query
        formatted_query = f"""
        Medical Question: {question}
        
        Background Context: {background}
        
        Research Objective: {objective}
        
        Based on the available evidence, does this intervention provide benefits, cause harm, or is there insufficient information to determine the effect?
        
        Please provide a clear verdict: YES (beneficial), NO (harmful), or NOT ENOUGH INFORMATION (insufficient evidence).
        """
        
        return formatted_query.strip()
    
    @observe()
    def query_rag_system(self, formatted_question: str) -> str:
        """Query RAG system and return response."""
        if self.query_engine is None:
            raise ValueError("RAG system not initialized. Call setup_rag_system() first.")
        
        try:
            response = self.query_engine.query(formatted_question)
            return str(response.response)
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return "ERROR: Could not generate response"
    
    @observe()
    def evaluate_single_question(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Evaluate a single question from MedREQAL dataset."""
        # Format question for RAG
        formatted_question = self.format_question_with_context(row)
        
        # Query RAG system
        rag_response = self.query_rag_system(formatted_question)
        
        # Extract verdict from response
        rag_verdict = extract_verdict_from_response(rag_response)
        
        # Get ground truth
        true_verdict = row['verdicts']
        
        # Prepare result
        result = {
            'index': index,
            'question': row['question'],
            'category': row.get('category', 'unknown'),
            'true_verdict': true_verdict,
            'rag_response': rag_response,
            'rag_verdict': rag_verdict,
            'background': row.get('background', ''),
            'conclusion': row.get('conclusion', ''),
            'strength': row.get('strength', ''),
            'label': row.get('label', '')
        }
        
        return result
    
    @observe()
    def evaluate_dataset(
        self, 
        csv_path: str, 
        output_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate complete MedREQAL dataset."""
        
        # Setup RAG system
        self.setup_rag_system()
        
        # Load data
        df = self.load_medreqal_data(csv_path)
        
        # Limit dataset if specified
        if limit:
            df = df.head(limit)
            print(f"Limited evaluation to {limit} questions")
        
        # Evaluate each question
        print(f"Starting evaluation of {len(df)} questions...")
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
            try:
                result = self.evaluate_single_question(row, index)
                results.append(result)
                
                # Add delay to avoid rate limiting
                if self.delay_between_queries > 0:
                    time.sleep(self.delay_between_queries)
                    
            except Exception as e:
                print(f"Error evaluating question {index}: {e}")
                # Add error result
                results.append({
                    'index': index,
                    'question': row['question'],
                    'category': row.get('category', 'unknown'),
                    'true_verdict': row['verdicts'],
                    'rag_response': f"ERROR: {str(e)}",
                    'rag_verdict': 'ERROR',
                    'error': str(e)
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate overall metrics
        valid_results = results_df[results_df['rag_verdict'] != 'ERROR']
        
        if len(valid_results) > 0:
            true_verdicts = valid_results['true_verdict'].tolist()
            rag_verdicts = valid_results['rag_verdict'].tolist()
            
            overall_metrics = evaluate_predictions(true_verdicts, rag_verdicts)
            category_metrics = calculate_category_performance(
                valid_results, 
                'true_verdict', 
                'rag_verdict', 
                'category'
            )
            
            metrics = {
                'overall': overall_metrics,
                'by_category': category_metrics,
                'total_questions': len(df),
                'successful_evaluations': len(valid_results),
                'failed_evaluations': len(results_df) - len(valid_results)
            }
        else:
            metrics = {'error': 'No successful evaluations'}
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        print("Evaluation completed!")
        return results_df, metrics
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("MEDREQAL EVALUATION SUMMARY")
        print("="*50)
        
        if 'error' in metrics:
            print(f"{metrics['error']}")
            return
        
        overall = metrics['overall']
        print("Overall Performance:")
        print(f"   Accuracy: {overall['accuracy']:.3f}")
        print(f"   F1 (Weighted): {overall['f1_weighted']:.3f}")
        print(f"   F1 (Macro): {overall['f1_macro']:.3f}")
        print(f"   Total Samples: {overall['total_samples']}")
        
        print("\nEvaluation Stats:")
        print(f"   Total Questions: {metrics['total_questions']}")
        print(f"   Successful: {metrics['successful_evaluations']}")
        print(f"   Failed: {metrics['failed_evaluations']}")
        
        print("\nPerformance by Category:")
        for category, cat_metrics in metrics['by_category'].items():
            print(f"   {category}:")
            print(f"     Accuracy: {cat_metrics['accuracy']:.3f}")
            print(f"     F1: {cat_metrics['f1_weighted']:.3f}")
            print(f"     Samples: {cat_metrics['sample_count']}")
        
        print("="*50)
