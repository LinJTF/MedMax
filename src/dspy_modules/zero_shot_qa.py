import dspy
from .signatures import ZeroShotQASignature


class ZeroShotQAModule(dspy.Module):
    """
    DSPy module for zero-shot medical question answering.
    
    Features:
    - Chain-of-Thought reasoning
    - Structured output without retrieval
    - Automatic validation
    """
    
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(ZeroShotQASignature)
    
    def forward(self, question: str):
        """
        Generate structured answer without retrieval.
        
        Args:
            question: Medical question to answer
        
        Returns:
            dspy.Prediction with:
                - verdict: str ('yes', 'no', or 'maybe')
                - reasoning: str (explanation)
        """
        try:
            prediction = self.generate_answer(question=question)
            
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
            print(f" DSPy Zero-Shot error: {e}")
            return dspy.Prediction(
                verdict='maybe',
                reasoning=f'Error generating answer: {str(e)}'
            )