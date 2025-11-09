import dspy
from typing import List, Union
from .signatures import MedicalQASignature


class MedicalRAGModule(dspy.Module):
    """
    DSPy module for answering medical questions using retrieved contexts.
    
    Features:
    - Chain-of-Thought reasoning
    - Structured output (verdict + reasoning)
    - Automatic validation and normalization
    """
    
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(MedicalQASignature)
    
    def forward(self, question: str, contexts: Union[List[str], str]):
        """
        Generate structured answer for medical question with contexts.
        
        Args:
            question: Medical question to answer
            contexts: Either a list of context strings or a single string
        
        Returns:
            dspy.Prediction with:
                - verdict: str ('yes', 'no', or 'maybe')
                - reasoning: str (explanation)
        """
        if isinstance(contexts, list):
            contexts_text = "\n".join([
                f"[{i+1}] {ctx.strip()}"
                for i, ctx in enumerate(contexts)
                if ctx and ctx.strip()
            ])
        else:
            contexts_text = contexts

        if not contexts_text or not contexts_text.strip():
            return dspy.Prediction(
                verdict='maybe',
                reasoning='No relevant research contexts were found to answer this question.'
            )

        try:
            prediction = self.generate_answer(
                question=question,
                contexts=contexts_text
            )

            verdict = str(prediction.verdict).lower().strip()

            if verdict not in ['yes', 'no', 'maybe']:
                reasoning_lower = str(prediction.reasoning).lower()
                if 'yes' in reasoning_lower[:50] or 'beneficial' in reasoning_lower:
                    verdict = 'yes'
                elif 'no' in reasoning_lower[:50] or 'harmful' in reasoning_lower:
                    verdict = 'no'
                else:
                    verdict = 'maybe'
            
            return dspy.Prediction(
                verdict=verdict,
                reasoning=prediction.reasoning
            )
            
        except Exception as e:
            print(f"DSPy RAG error: {e}")
            return dspy.Prediction(
                verdict='maybe',
                reasoning=f'Error processing contexts: {str(e)}'
            )
