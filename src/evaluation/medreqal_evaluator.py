"""MedREQAL evaluator for comparing RAG performance with zero-shot results."""

from langfuse import observe
from src.observability.langfuse_config import create_session, update_current_generation, update_trace_metadata, update_span_metadata

import pandas as pd
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from src.observability.token_cost import calculate_cost, get_token_usage

# RAG system imports
from ..rag.models import setup_llm
from ..rag.client import setup_rag_client
from ..rag.models import configure_global_settings
from ..rag.query_engine import create_simple_query_engine, create_standard_query_engine, create_enhanced_query_engine
from ..rag.main import patch_query_engine_with_tracing
from .metrics import evaluate_predictions, extract_verdict_from_response, extract_pubmedqa_verdict_from_response, calculate_category_performance


class MedREQALEvaluator:
    """Evaluator for MedREQAL and PubMedQA datasets using RAG system."""
    
    def __init__(
        self,
        collection_name: str = "medmax_pubmed",
        engine_type: str = "standard",  # Changed default to standard
        delay_between_queries: float = 1.0,
        mode: str = "rag",
        llm_model: str = "gpt-4o-mini",
        dataset_type: str = "medreqal",  # New parameter
        use_ollama: bool = False,  # New parameter for Ollama support
    ):
        """Initialize evaluator."""
        self.collection_name = collection_name
        self.engine_type = engine_type
        self.delay_between_queries = delay_between_queries
        self.query_engine = None
        self.results = []
        self.mode = mode
        self.llm_model = llm_model
        self.dataset_type = dataset_type  # Track which dataset we're evaluating
        self.use_ollama = use_ollama  # Store Ollama preference
        
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
        
        # Apply tracing patch to query engine to capture all LLM calls
        self.query_engine = patch_query_engine_with_tracing(self.query_engine)
        
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
        
        update_span_metadata({
            "operation": "rag_query",
            "formatted_question": formatted_question,
            "collection_name": self.collection_name,
            "engine_type": self.engine_type
        })
        
        if self.query_engine is None:
            raise ValueError("RAG system not initialized. Call setup_rag_system() first.")
        
        try:
            response = self.query_engine.query(formatted_question)
            response_text = str(response.response) if hasattr(response, "response") else str(response)
                           
            token_usage = get_token_usage(formatted_question, response_text, self.llm_model)
            cost = calculate_cost(
                token_usage['input_tokens'],
                token_usage['output_tokens'],
                self.llm_model
            )
            
            update_current_generation(
                model=self.llm_model,
                input_text=formatted_question,
                output_text=response_text,
                input_tokens=token_usage["input_tokens"],
                output_tokens=token_usage["output_tokens"],
                input_cost=cost["input_cost"],
                output_cost=cost["output_cost"],
                metadata={
                    "evaluation_mode": "rag",
                    "engine_type": self.engine_type,
                    "collection_name": self.collection_name,
                }
            )
            return str(response.response)
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return "ERROR: Could not generate response"
    
    @observe()
    def zero_shot_system(self, prompt: str, question_index: int, category: str) -> Tuple[str, str]:
        """Perform zero-shot LLM call with proper generation tracking."""
        update_span_metadata({
            "operation": "zero_shot_generation",
            "question_index": question_index,
            "category": category,
            "model": self.llm_model
        })
        
        try:
            llm = setup_llm(model=self.llm_model, use_ollama=self.use_ollama)
            response = llm.complete(prompt)
            answer = response.text if hasattr(response, "text") else str(response)
            verdict = extract_verdict_from_response(answer)
            
            token_usage = get_token_usage(prompt, answer, self.llm_model)
            costs = calculate_cost(
                token_usage['input_tokens'],
                token_usage['output_tokens'],
                self.llm_model
            )

            update_current_generation(
                model=self.llm_model,
                input_text=prompt,
                output_text=answer,
                input_tokens=token_usage["input_tokens"],
                output_tokens=token_usage["output_tokens"],
                input_cost=costs["input_cost"],
                output_cost=costs["output_cost"],
                metadata={
                    "evaluation_mode": "zero_shot",
                    "engine_type": self.engine_type,
                    "collection_name": self.collection_name,
                }
            )
            
            return answer, verdict
            
        except Exception as e:
            print(f"Error in zero-shot LLM call: {e}")
            return f"ERROR: {e}", "ERROR"
    
    @observe()
    def evaluate_single_question(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Evaluate a single question from MedREQAL dataset."""
        # Perform zero-shot evaluation if mode is set to 'zero_shot'
        update_span_metadata({
            "operation": "evaluate_single_question",
            "question_index": index,
            "mode": self.mode,
        })
        if self.mode == "zero_shot":
            prompt = (
                f"Medical Question: {row['question']}\n"
                "Please answer and provide a clear verdict: SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION."
            )
            answer, verdict = self.zero_shot_system(
                prompt, 
                question_index=index, 
                category=row.get('category', 'unknown')
            )
    
            result = {
                'index': index,
                'question': row['question'],
                'category': row.get('category', 'unknown'),
                'true_verdict': row['verdicts'],
                'llm_response': answer,
                'llm_verdict': verdict,
                'mode': self.mode
            }
        
        # Format question for RAG
        else:
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
                'label': row.get('label', ''),
                'mode': self.mode
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
        update_trace_metadata({
            "operation": "evaluate_dataset",
            "mode": self.mode,
            "service": "medreqal_evaluation",
            "engine_type": self.engine_type,
            "collection_name": self.collection_name,
            "llm_model": self.llm_model,
            "dataset_path": csv_path,
            "limit": limit
        })
        # Setup system
        if self.mode == "zero_shot":
            print("Running in ZERO-SHOT mode (no retrieval)...")
        else:
            self.setup_rag_system()

        # Load data
        df = self.load_medreqal_data(csv_path)
        if limit:
            df = df.head(limit)
            print(f"Limited evaluation to {limit} questions")

        print(f"Starting evaluation of {len(df)} questions...")
        results = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
            try:
                result = self.evaluate_single_question(row, index)
                results.append(result)
                if self.delay_between_queries > 0:
                    time.sleep(self.delay_between_queries)
            except Exception as e:
                print(f"Error evaluating question {index}: {e}")
                # Add error result
                error_result = {
                    'index': index,
                    'question': row['question'],
                    'category': row.get('category', 'unknown'),
                    'true_verdict': row['verdicts'],
                    'error': str(e)
                }
                if self.mode == "zero_shot":
                    error_result['llm_response'] = f"ERROR: {str(e)}"
                    error_result['llm_verdict'] = "ERROR"
                else:
                    error_result['rag_response'] = f"ERROR: {str(e)}"
                    error_result['rag_verdict'] = "ERROR"
                results.append(error_result)

        results_df = pd.DataFrame(results)
        # Metrics
        if self.mode == "zero_shot":
            valid_results = results_df[results_df['llm_verdict'] != 'ERROR']
            true_verdicts = valid_results['true_verdict'].tolist()
            pred_verdicts = valid_results['llm_verdict'].tolist()
            overall_metrics = evaluate_predictions(true_verdicts, pred_verdicts)
            category_metrics = calculate_category_performance(
                valid_results, 
                'true_verdict', 
                'llm_verdict', 
                'category'
            )
        else:
            valid_results = results_df[results_df['rag_verdict'] != 'ERROR']
            true_verdicts = valid_results['true_verdict'].tolist()
            pred_verdicts = valid_results['rag_verdict'].tolist()
            overall_metrics = evaluate_predictions(true_verdicts, pred_verdicts)
            category_metrics = calculate_category_performance(
                valid_results, 
                'true_verdict', 
                'rag_verdict', 
                'category'
            )
        if len(valid_results) > 0:
            metrics = {
                'overall': overall_metrics,
                'by_category': category_metrics,
                'total_questions': len(df),
                'successful_evaluations': len(valid_results),
                'failed_evaluations': len(results_df) - len(valid_results)
            }
        else:
            metrics = {'error': 'No successful evaluations'}

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

    # PubMedQA-specific methods
    def load_pubmedqa_data(self, parquet_path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load PubMedQA dataset from parquet file."""
        print(f"Loading PubMedQA data from {parquet_path}")
        
        try:
            df = pd.read_parquet(parquet_path)
            if limit:
                df = df.head(limit)
                print(f"Limited to {limit} questions")
            print(f"Loaded {len(df)} questions from PubMedQA dataset")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load parquet file: {e}")
    
    def format_pubmedqa_question_with_context(self, row: pd.Series) -> str:
        """Format PubMedQA question with context for RAG query."""
        question = row['question']
        # Extract context from the context column - it's a dict with 'contexts' array
        context_data = row['context'] if isinstance(row['context'], dict) else {}
        contexts = context_data.get('contexts', [])
        
        # Handle numpy arrays properly
        if hasattr(contexts, 'tolist'):
            contexts = contexts.tolist()
        
        context_text = " ".join(str(ctx) for ctx in contexts) if contexts else ""
        
        formatted_query = f"""
        Medical Question: {question}
        
        Available Evidence: {context_text}
        
        Based on the available evidence (which is the context above), does this intervention or condition provide benefits, cause harm, or is there insufficient information to determine the effect?
        
        Please provide a clear verdict: YES (beneficial/supported), NO (harmful/not supported), or MAYBE (insufficient evidence).
        """
        
        return formatted_query.strip()
    
    @observe()
    def evaluate_single_pubmedqa_question(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Evaluate a single question from PubMedQA dataset."""
        update_span_metadata({
            "operation": "evaluate_single_pubmedqa_question",
            "question_index": index,
            "mode": self.mode,
            "dataset_type": "pubmedqa"
        })
        
        if self.mode == "zero_shot":
            prompt = (
                f"Medical Question: {row['question']}\n"
                "Please answer and provide a clear verdict: YES, NO, or MAYBE."
            )
            
            try:
                llm = setup_llm(model=self.llm_model, use_ollama=self.use_ollama)
                response = llm.complete(prompt)
                answer = response.text if hasattr(response, "text") else str(response)
                # Use PubMedQA-specific verdict extraction
                verdict = extract_pubmedqa_verdict_from_response(answer)
                
                # Add cost tracking for PubMedQA zero-shot evaluation
                token_usage = get_token_usage(prompt, answer, self.llm_model)
                costs = calculate_cost(
                    token_usage['input_tokens'],
                    token_usage['output_tokens'],
                    self.llm_model
                )

                update_current_generation(
                    model=self.llm_model,
                    input_text=prompt,
                    output_text=answer,
                    input_tokens=token_usage["input_tokens"],
                    output_tokens=token_usage["output_tokens"],
                    input_cost=costs["input_cost"],
                    output_cost=costs["output_cost"],
                    metadata={
                        "evaluation_mode": "zero_shot",
                        "dataset_type": "pubmedqa",
                        "engine_type": self.engine_type,
                        "collection_name": self.collection_name,
                    }
                )
                
            except Exception as e:
                print(f"Error in zero-shot generation: {e}")
                answer = "ERROR"
                verdict = "ERROR"
    
            result = {
                'index': index,
                'pubid': row['pubid'],
                'question': row['question'],
                'true_verdict': row['final_decision'],
                'llm_response': answer,
                'llm_verdict': verdict,
                'mode': self.mode,
                'dataset_type': 'pubmedqa'
            }
        else:
            # RAG mode
            formatted_question = self.format_pubmedqa_question_with_context(row)
            
            # Query RAG system
            rag_response = self.query_rag_system(formatted_question)
            
            # Extract verdict from response using PubMedQA-specific function
            rag_verdict = extract_pubmedqa_verdict_from_response(rag_response)
            
            # Get ground truth
            true_verdict = row['final_decision']
            
            # Prepare result
            result = {
                'index': index,
                'pubid': row['pubid'],
                'question': row['question'],
                'true_verdict': true_verdict,
                'rag_response': rag_response,
                'rag_verdict': rag_verdict,
                'long_answer': row.get('long_answer', ''),
                'mode': self.mode,
                'dataset_type': 'pubmedqa'
            }
        
        return result
    
    @observe()
    def evaluate_pubmedqa_dataset(
            self, 
            parquet_path: str, 
            output_path: Optional[str] = None,
            limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate complete PubMedQA dataset."""
        update_trace_metadata({
            "operation": "evaluate_pubmedqa_dataset",
            "mode": self.mode,
            "service": "pubmedqa_evaluation",
            "engine_type": self.engine_type,
            "collection_name": self.collection_name,
            "llm_model": self.llm_model,
            "dataset_path": parquet_path,
            "dataset_type": "pubmedqa",
            "limit": limit
        })
        
        # Setup system
        if self.mode == "zero_shot":
            print("Running PubMedQA evaluation in ZERO-SHOT mode (no retrieval)...")
        else:
            self.setup_rag_system()

        # Load data
        df = self.load_pubmedqa_data(parquet_path, limit)

        print(f"Starting PubMedQA evaluation of {len(df)} questions...")
        results = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating PubMedQA questions"):
            try:
                result = self.evaluate_single_pubmedqa_question(row, index)
                results.append(result)
                if self.delay_between_queries > 0:
                    time.sleep(self.delay_between_queries)
            except Exception as e:
                print(f"Error evaluating PubMedQA question {index}: {e}")
                # Add error result
                error_result = {
                    'index': index,
                    'pubid': row['pubid'],
                    'question': row['question'],
                    'true_verdict': row['final_decision'],
                    'error': str(e),
                    'dataset_type': 'pubmedqa'
                }
                if self.mode == "zero_shot":
                    error_result['llm_response'] = f"ERROR: {str(e)}"
                    error_result['llm_verdict'] = "ERROR"
                else:
                    error_result['rag_response'] = f"ERROR: {str(e)}"
                    error_result['rag_verdict'] = "ERROR"
                results.append(error_result)

        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        if self.mode == "zero_shot":
            valid_results = results_df[results_df['llm_verdict'] != 'ERROR']
            true_verdicts = valid_results['true_verdict'].tolist()
            pred_verdicts = valid_results['llm_verdict'].tolist()
        else:
            valid_results = results_df[results_df['rag_verdict'] != 'ERROR']
            true_verdicts = valid_results['true_verdict'].tolist()
            pred_verdicts = valid_results['rag_verdict'].tolist()
        
        if len(valid_results) > 0:
            overall_metrics = evaluate_predictions(true_verdicts, pred_verdicts)
            metrics = {
                'overall': overall_metrics,
                'total_questions': len(df),
                'successful_evaluations': len(valid_results),
                'failed_evaluations': len(results_df) - len(valid_results),
                'dataset_type': 'pubmedqa'
            }
        else:
            metrics = {'error': 'No successful evaluations', 'dataset_type': 'pubmedqa'}

        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"PubMedQA results saved to {output_path}")

        print("PubMedQA evaluation completed!")
        return results_df, metrics
    
    def print_pubmedqa_summary(self, metrics: Dict[str, Any]):
        """Print PubMedQA evaluation summary."""
        print("\n" + "="*50)
        print("PUBMEDQA EVALUATION SUMMARY")
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
        
        print("="*50)
