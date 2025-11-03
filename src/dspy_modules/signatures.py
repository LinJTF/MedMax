import dspy

class MedicalQASignature(dspy.Signature):
    """
    Signature for RAG-based medical question answering.
    
    Given a medical question and retrieved research contexts from PubMed,
    provide a structured verdict with reasoning.
    """

    question = dspy.InputField(
        desc="Medical question to answer"
    )
    contexts = dspy.InputField(
        desc="Retrieved research contexts from PubMed that may contain relevant evidence"
    )

    verdict = dspy.OutputField(
        desc="Final verdict: 'yes' (beneficial/supported), 'no' (harmful/refuted), or 'maybe' (insufficient evidence)"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation citing specific evidence from the provided contexts"
    )


class ZeroShotQASignature(dspy.Signature):
    """
    Signature for zero-shot medical question answering (no retrieval).
    
    Answer medical questions based solely on the model's knowledge.
    """
    question = dspy.InputField(
        desc="Medical question to answer"
    )
    
    verdict = dspy.OutputField(
        desc="Final verdict: 'yes' (beneficial/supported), 'no' (harmful/refuted), or 'maybe' (insufficient evidence)"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of the answer based on medical knowledge"
    )