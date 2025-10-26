"""LLM models configuration for RAG system (OpenAI + Ollama)."""

import os
from typing import Optional, Union
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

load_dotenv()

OLLAMA_MODELS = {
    "mistral:7b": {
        "name": "mistral:7b",
        "context_window": 32768,
        "description": "Mistral 7B - Fast and efficient (Recommended)"
    },
    "qwen2.5:7b": {
        "name": "qwen2.5:7b",
        "context_window": 32768,
        "description": "Qwen 2.5 7B - Strong analytical capabilities"
    }
}


def setup_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    use_ollama: bool = False,
    ollama_base_url: str = "http://localhost:11434"
) -> Union[OpenAI, Ollama]:
    """
    Setup LLM for RAG responses (OpenAI or Ollama).
    
    Args:
        model: Model name (e.g., "gpt-4o-mini" or "llama3.1:8b")
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        use_ollama: If True, use Ollama instead of OpenAI
        ollama_base_url: Ollama server URL (default: http://localhost:11434)
    
    Returns:
        Configured LLM instance (OpenAI or Ollama)
    """
    if use_ollama:
        # Use Ollama for local models
        if model in OLLAMA_MODELS:
            model_name = OLLAMA_MODELS[model]["name"]
            context_window = OLLAMA_MODELS[model]["context_window"]
            print(f"[Ollama] Using model: {model_name}")
            print(f"   {OLLAMA_MODELS[model]['description']}")
        else:
            model_name = model
            context_window = 4096
            print(f"[Warning] Using custom Ollama model: {model_name}")
        
        llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            context_window=context_window,
            request_timeout=120.0
        )
        
        print(f"[OK] Ollama LLM configured: {model_name} (temp: {temperature})")
        return llm
    else:
        # Use OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        llm = OpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=120
        )
        
        print(f"[OK] OpenAI LLM configured: {model} (temp: {temperature})")
        return llm


def setup_embedding_model(model: str = "text-embedding-3-small") -> OpenAIEmbedding:
    """Setup OpenAI embedding model for queries."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    embed_model = OpenAIEmbedding(
        model=model,
        api_key=api_key
    )
    
    print(f"Embedding model configured: {model}")
    return embed_model


def configure_global_settings():
    """Configure LlamaIndex global settings."""
    Settings.llm = setup_llm()
    Settings.embed_model = setup_embedding_model()
    print("Global LlamaIndex settings configured")
