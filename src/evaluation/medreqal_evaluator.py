"""MedREQAL evaluator - OPTIMIZED VERSION (loads LLM once)."""

from langfuse import observe
from src.observability.langfuse_config import update_current_generation, update_trace_metadata, update_span_metadata

import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from tqdm import tqdm

from src.observability.token_cost import calculate_cost, get_token_usage

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
        engine_type: str = "standard",
        delay_between_queries: float = 1.0,
        mode: str = "rag",
        llm_model: str = "gpt-4o-mini",
        dataset_type: str = "medreqal",
        use_ollama: bool = False,
        use_huggingface: bool = False
    ):
        """Initialize evaluator."""
        print(f"[Evaluator.__init__] Initializing with mode={mode}, model={llm_model}")
        
        self.collection_name = collection_name
        self.engine_type = engine_type
        self.delay_between_queries = delay_between_queries
        self.query_engine = None
        self.mode = mode
        self.llm_model = llm_model
        self.dataset_type = dataset_type
        self.use_ollama = use_ollama
        self.use_huggingface = use_huggingface
        
        self.llm = None
        if self.mode == "zero_shot":
            print(f"\n{'='*70}")
            print(f" INITIALIZING {llm_model.upper()} FOR ZERO-SHOT EVALUATION")
            print(f"{'='*70}")
            print(" Loading model (this may take a while for HuggingFace)...")
            
            self.llm = setup_llm(
                model=self.llm_model,
                use_ollama=self.use_ollama,
                use_huggingface=self.use_huggingface
            )
            
            print(" LLM loaded successfully and ready!")
            print(f"{'='*70}\n")
        
        print(f"[Evaluator.__init__] Initialization complete")
    
    def setup_rag_system(self):
        """Setup RAG system for evaluation."""
        print(f"Setting up RAG system with model={self.llm_model}, ollama={self.use_ollama}, hf={self.use_huggingface}...")
        configure_global_settings(
            llm_model=self.llm_model,
            use_ollama=self.use_ollama,
            use_huggingface=self.use_huggingface
        )
        _, index = setup_rag_client(self.collection_name)
        
        if self.engine_type == "simple":
            self.query_engine = create_simple_query_engine(index, top_k=5)
        elif self.engine_type == "enhanced":
            self.query_engine = create_enhanced_query_engine(
                index, collection_name=self.collection_name, top_k=5
            )
        else:
            self.query_engine = create_standard_query_engine(
                index, collection_name=self.collection_name, top_k=5
            )
        
        self.query_engine = patch_query_engine_with_tracing(self.query_engine)
        print("RAG system ready")
    
    def load_medreqal_data(self, csv_path: str) -> pd.DataFrame:
        """Load MedREQAL dataset."""
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} questions from MedREQAL")
        return df
    
    def load_pubmedqa_data(self, parquet_path: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load PubMedQA dataset."""
        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.head(limit)
        print(f"Loaded {len(df)} questions from PubMedQA")
        return df
    
    def format_question_with_context(self, row: pd.Series) -> str:
        """Format MedREQAL question with context."""
        return f"""
Medical Question: {row['question']}
Background: {row.get('background', '')}
Objective: {row.get('objective', '')}

Based on evidence, does this provide benefits, cause harm, or insufficient info?
Verdict: YES (beneficial), NO (harmful), or NOT ENOUGH INFORMATION.
        """.strip()
    
    def format_pubmedqa_question_with_context(self, row: pd.Series) -> str:
        """Format PubMedQA question with context."""
        context_data = row['context'] if isinstance(row['context'], dict) else {}
        contexts = context_data.get('contexts', [])
        if hasattr(contexts, 'tolist'):
            contexts = contexts.tolist()
        context_text = " ".join(str(ctx) for ctx in contexts) if contexts else ""
        
        return f"""
Medical Question: {row['question']}
Available Evidence: {context_text}

Based on evidence, does this provide benefits, cause harm, or insufficient info?
Verdict: YES (beneficial), NO (harmful), or MAYBE (insufficient).
        """.strip()
    
    @observe()
    def query_rag_system(self, formatted_question: str) -> str:
        """Query RAG system."""
        if self.query_engine is None:
            raise ValueError("RAG system not initialized")
        
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
                metadata={"evaluation_mode": "rag"}
            )
            
            return response_text
        except Exception as e:
            print(f"Error querying RAG: {e}")
            return f"ERROR: {e}"
    
    @observe()
    def zero_shot_query(self, prompt: str, question_index: int) -> Tuple[str, str]:
        """Perform zero-shot query using pre-initialized LLM (OPTIMIZED)."""
        
        if self.llm is None:
            raise ValueError("LLM not initialized for zero-shot mode")
        
        try:
            response = self.llm.complete(prompt)
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
                metadata={"evaluation_mode": "zero_shot", "question_index": question_index}
            )
            
            return answer, verdict
            
        except Exception as e:
            print(f" Error in zero-shot query: {e}")
            return f"ERROR: {e}", "ERROR"
    
    @observe()
    def zero_shot_query_pubmedqa(self, prompt: str, question_index: int) -> Tuple[str, str]:
        """Zero-shot query for PubMedQA (uses different verdict extraction)."""
        
        if self.llm is None:
            raise ValueError("LLM not initialized for zero-shot mode")
        
        try:
            response = self.llm.complete(prompt)
            answer = response.text if hasattr(response, "text") else str(response)
            verdict = extract_pubmedqa_verdict_from_response(answer)
            
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
                metadata={"evaluation_mode": "zero_shot", "question_index": question_index}
            )
            
            return answer, verdict
            
        except Exception as e:
            print(f" Error in zero-shot query: {e}")
            return f"ERROR: {e}", "ERROR"
    
    @observe()
    def evaluate_single_question(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Evaluate single MedREQAL question."""
        if self.mode == "zero_shot":
            prompt = f"Medical Question: {row['question']}\nVerdict: SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION."
            answer, verdict = self.zero_shot_query(prompt, index)
            
            return {
                'index': index,
                'question': row['question'],
                'category': row.get('category', 'unknown'),
                'true_verdict': row['verdicts'],
                'llm_response': answer,
                'llm_verdict': verdict,
                'mode': self.mode
            }
        else:
            formatted_question = self.format_question_with_context(row)
            rag_response = self.query_rag_system(formatted_question)
            rag_verdict = extract_verdict_from_response(rag_response)
            
            return {
                'index': index,
                'question': row['question'],
                'category': row.get('category', 'unknown'),
                'true_verdict': row['verdicts'],
                'rag_response': rag_response,
                'rag_verdict': rag_verdict,
                'mode': self.mode
            }
    
    @observe()
    def evaluate_single_pubmedqa_question(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Evaluate single PubMedQA question (OPTIMIZED)."""
        if self.mode == "zero_shot":
            prompt = f"Medical Question: {row['question']}\nVerdict: YES, NO, or MAYBE."
            answer, verdict = self.zero_shot_query_pubmedqa(prompt, index)
            
            return {
                'index': index,
                'pubid': row['pubid'],
                'question': row['question'],
                'true_verdict': row['final_decision'],
                'llm_response': answer,
                'llm_verdict': verdict,
                'mode': self.mode,
                'dataset_type': 'pubmedqa',
                'model_name': self.llm_model
            }
        else:
            formatted_question = self.format_pubmedqa_question_with_context(row)
            rag_response = self.query_rag_system(formatted_question)
            rag_verdict = extract_pubmedqa_verdict_from_response(rag_response)
            
            return {
                'index': index,
                'pubid': row['pubid'],
                'question': row['question'],
                'true_verdict': row['final_decision'],
                'rag_response': rag_response,
                'rag_verdict': rag_verdict,
                'mode': self.mode,
                'dataset_type': 'pubmedqa',
                'model_name': self.llm_model
            }
    
    @observe()
    def evaluate_dataset(
        self, 
        csv_path: str, 
        output_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate MedREQAL dataset."""
        if self.mode != "zero_shot":
            self.setup_rag_system()
        
        df = self.load_medreqal_data(csv_path)
        if limit:
            df = df.head(limit)
            print(f"Limited to {limit} questions")
        
        print(f"Starting evaluation of {len(df)} questions...")
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                result = self.evaluate_single_question(row, index)
                results.append(result)
                if self.delay_between_queries > 0:
                    time.sleep(self.delay_between_queries)
            except Exception as e:
                print(f" Error on question {index}: {e}")
                results.append({
                    'index': index,
                    'question': row['question'],
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        verdict_col = 'llm_verdict' if self.mode == "zero_shot" else 'rag_verdict'
        valid_results = results_df[results_df[verdict_col] != 'ERROR']
        
        if len(valid_results) > 0:
            metrics = {
                'overall': evaluate_predictions(
                    valid_results['true_verdict'].tolist(),
                    valid_results[verdict_col].tolist()
                ),
                'by_category': calculate_category_performance(
                    valid_results, 'true_verdict', verdict_col, 'category'
                ),
                'total_questions': len(df),
                'successful_evaluations': len(valid_results),
                'failed_evaluations': len(results_df) - len(valid_results)
            }
        else:
            metrics = {'error': 'No successful evaluations'}
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f" Results saved to {output_path}")
        
        return results_df, metrics
    
    @observe()
    def evaluate_pubmedqa_dataset(
        self,
        parquet_path: str,
        output_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate PubMedQA dataset."""
        if self.mode != "zero_shot":
            self.setup_rag_system()
        
        df = self.load_pubmedqa_data(parquet_path, limit)
        
        print(f"Starting PubMedQA evaluation of {len(df)} questions...")
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating PubMedQA"):
            try:
                result = self.evaluate_single_pubmedqa_question(row, index)
                results.append(result)
                if self.delay_between_queries > 0:
                    time.sleep(self.delay_between_queries)
            except Exception as e:
                print(f" Error on question {index}: {e}")
                results.append({
                    'index': index,
                    'pubid': row['pubid'],
                    'question': row['question'],
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        verdict_col = 'llm_verdict' if self.mode == "zero_shot" else 'rag_verdict'
        valid_results = results_df[results_df[verdict_col] != 'ERROR']
        
        if len(valid_results) > 0:
            metrics = {
                'overall': evaluate_predictions(
                    valid_results['true_verdict'].tolist(),
                    valid_results[verdict_col].tolist()
                ),
                'total_questions': len(df),
                'successful_evaluations': len(valid_results),
                'failed_evaluations': len(results_df) - len(valid_results),
                'dataset_type': 'pubmedqa',
                'model_name': self.llm_model
            }
        else:
            metrics = {'error': 'No successful evaluations', 'dataset_type': 'pubmedqa'}
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f" Results saved to {output_path}")
        
        return results_df, metrics
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print MedREQAL summary."""
        print("\n" + "="*50)
        print("MEDREQAL EVALUATION SUMMARY")
        print("="*50)
        
        if 'error' in metrics:
            print(f" {metrics['error']}")
            return
        
        overall = metrics['overall']
        print(f"Accuracy: {overall['accuracy']:.3f}")
        print(f"F1 (Weighted): {overall['f1_weighted']:.3f}")
        print(f"F1 (Macro): {overall['f1_macro']:.3f}")
        print(f"Total: {metrics['total_questions']}")
        print(f"Successful: {metrics['successful_evaluations']}")
        print(f"Failed: {metrics['failed_evaluations']}")
        print("="*50)
    
    def print_pubmedqa_summary(self, metrics: Dict[str, Any]):
        """Print PubMedQA summary."""
        print("\n" + "="*50)
        print("PUBMEDQA EVALUATION SUMMARY")
        print("="*50)
        
        if 'error' in metrics:
            print(f" {metrics['error']}")
            return
        
        overall = metrics['overall']
        print(f"Accuracy: {overall['accuracy']:.3f}")
        print(f"F1 (Weighted): {overall['f1_weighted']:.3f}")
        print(f"F1 (Macro): {overall['f1_macro']:.3f}")
        print(f"Total: {metrics['total_questions']}")
        print(f"Successful: {metrics['successful_evaluations']}")
        print(f"Failed: {metrics['failed_evaluations']}")
        print("="*50)
