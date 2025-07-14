"""Query engine setup for RAG system."""

from typing import Optional, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.prompts import PromptTemplate

from .retriever import QdrantRetriever
from .models import setup_llm


# Custom prompt for medical queries
MEDICAL_QA_PROMPT = PromptTemplate(
    """You are a medical AI assistant specialized in answering questions based on PubMed research data.

Context Information:
{context_str}

Query: {query_str}

Instructions:
1. Answer the medical question based ONLY on the provided context from PubMed research
2. If the context doesn't contain relevant information, clearly state that
3. Always cite which context/study supports your answer
4. Be precise and evidence-based in your response
5. Include any relevant limitations or caveats mentioned in the research
6. Format your response clearly with sections if appropriate

Answer:"""
)


def create_query_engine(
    index: VectorStoreIndex,
    retriever: Optional[QdrantRetriever] = None,
    top_k: int = 5,
    response_mode: str = "compact",
    llm_model: str = "gpt-4o-mini",
    **kwargs: Any
) -> RetrieverQueryEngine:
    """Create a query engine for medical Q&A."""
    
    # Setup LLM
    llm = setup_llm(model=llm_model)
    
    # Use custom retriever if provided, otherwise use index retriever
    if retriever is not None:
        print(f"🔧 Using custom Qdrant retriever")
        final_retriever = retriever
    else:
        print(f"🔧 Using index retriever with top_k={top_k}")
        final_retriever = index.as_retriever(similarity_top_k=top_k)
    
    # Setup response synthesizer with custom prompt
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT if response_mode == "compact" else ResponseMode.TREE_SUMMARIZE,
        text_qa_template=MEDICAL_QA_PROMPT,
        llm=llm,
    )
    
    # Create query engine
    query_engine = RetrieverQueryEngine(
        retriever=final_retriever,
        response_synthesizer=response_synthesizer,
    )
    
    print(f"🎯 Query engine created with {llm_model} and {response_mode} mode")
    return query_engine


def create_standard_query_engine(
    index: VectorStoreIndex,
    collection_name: str = "medmax_pubmed",
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini",
    score_threshold: float = 0.7,
    **kwargs: Any
) -> RetrieverQueryEngine:
    """Create a standard query engine with custom Qdrant retriever for better metadata preservation."""
    
    # Import here to avoid circular imports
    from .client import setup_qdrant_client
    from .retriever import create_custom_retriever
    
    # Setup LLM
    llm = setup_llm(model=llm_model)
    
    # Setup custom Qdrant retriever
    qdrant_client = setup_qdrant_client()
    custom_retriever = create_custom_retriever(
        qdrant_client,
        collection_name,
        top_k=top_k,
        score_threshold=score_threshold
    )
    
    # Setup response synthesizer with medical prompt
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=MEDICAL_QA_PROMPT,
        llm=llm,
    )
    
    # Create query engine with custom retriever
    query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )
    
    print(f"🎯 Standard query engine created with custom retriever (top_k={top_k}, threshold={score_threshold})")
    return query_engine


def create_simple_query_engine(
    index: VectorStoreIndex,
    **kwargs: Any
) -> Any:
    """Create a simple query engine for quick testing."""
    # Setup LLM
    llm = setup_llm()
    
    # Create simple query engine with proper configurations
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=MEDICAL_QA_PROMPT,
        similarity_top_k=kwargs.get('top_k', 5),
        response_mode="compact",
        verbose=kwargs.get('verbose', False),
        **{k: v for k, v in kwargs.items() if k not in ['top_k', 'verbose']}
    )
    
    print("🚀 Simple query engine created")
    return query_engine


def create_enhanced_query_engine(
    index: VectorStoreIndex,
    collection_name: str = "medmax_pubmed",
    top_k: int = 10,
    llm_model: str = "gpt-4o-mini",
    score_threshold: float = 0.6,
    response_mode: str = "tree_summarize",
    custom_prompt: Optional[str] = None,
    **kwargs: Any
) -> RetrieverQueryEngine:
    """Create enhanced query engine with advanced configurations and custom retriever."""
    
    # Import here to avoid circular imports
    from .client import setup_qdrant_client
    from .retriever import create_custom_retriever
    
    # Setup LLM
    llm = setup_llm(model=llm_model)
    
    # Setup custom Qdrant retriever with more lenient threshold
    qdrant_client = setup_qdrant_client()
    custom_retriever = create_custom_retriever(
        qdrant_client,
        collection_name,
        top_k=top_k,
        score_threshold=score_threshold
    )
    
    # Use custom prompt if provided
    prompt_template = MEDICAL_QA_PROMPT
    if custom_prompt:
        prompt_template = PromptTemplate(custom_prompt)
    
    # Setup response synthesizer with advanced mode
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE if response_mode == "tree_summarize" else ResponseMode.COMPACT,
        text_qa_template=prompt_template,
        llm=llm,
    )
    
    # Create enhanced query engine
    query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )
    
    print(f"⚡ Enhanced query engine created (top_k={top_k}, threshold={score_threshold}, mode={response_mode})")
    return query_engine


def enhanced_query_engine(
    index: VectorStoreIndex,
    custom_prompt: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """Create enhanced query engine with custom configurations (legacy compatibility)."""
    
    # Use custom prompt if provided
    prompt_template = MEDICAL_QA_PROMPT
    if custom_prompt:
        prompt_template = PromptTemplate(custom_prompt)
    
    # Setup with advanced configurations
    query_engine = index.as_query_engine(
        text_qa_template=prompt_template,
        similarity_top_k=kwargs.get('top_k', 5),
        response_mode=kwargs.get('response_mode', 'compact'),
        verbose=kwargs.get('verbose', True),
    )
    
    print("⚡ Enhanced query engine created with custom configurations (legacy)")
    return query_engine
