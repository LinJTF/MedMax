"""OpenAI models configuration for RAG system."""

import os
from typing import Optional
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

load_dotenv()


def setup_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None
) -> OpenAI:
    """Setup OpenAI LLM for RAG responses."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    llm = OpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    
    print(f"🤖 LLM configured: {model} (temp: {temperature})")
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
    
    print(f"🔤 Embedding model configured: {model}")
    return embed_model


def configure_global_settings():
    """Configure LlamaIndex global settings."""
    Settings.llm = setup_llm()
    Settings.embed_model = setup_embedding_model()
    print("⚙️ Global LlamaIndex settings configured")
