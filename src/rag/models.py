"""LLM models configuration for RAG system (OpenAI + Ollama + HuggingFace)."""

import os
import torch
from typing import Optional, Union
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

HUGGINGFACE_MODELS = {
    "mistral-7b-instruct": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "context_window": 32768,
        "max_new_tokens": 512,
        "description": "Mistral 7B Instruct - HuggingFace (Local GPU)"
    },
    "qwen2.5-7b-instruct": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "context_window": 32768,
        "max_new_tokens": 512,
        "description": "Qwen 2.5 7B Instruct - HuggingFace (Local GPU)"
    },
    "llama3-8b-instruct": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "context_window": 8192,
        "max_new_tokens": 512,
        "description": "Llama 3 8B Instruct - HuggingFace (Local GPU)"
    }
}

HUGGINGFACE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "All MiniLM L6 v2 - Lightweight embedding model",
        "embed_dim": 384,
        "max_length": 256,
    }
}

def setup_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    use_ollama: bool = False,
    use_huggingface: bool = False,
    ollama_base_url: str = "http://localhost:11434",
    device: str = "auto"
) -> Union[OpenAI, Ollama, HuggingFaceLLM]:
    """
    Setup LLM for RAG responses (OpenAI, Ollama, or HuggingFace).
    
    Args:
        model: Model name (e.g., "gpt-4o-mini", "mistral:7b", or "mistral-7b-instruct")
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        use_ollama: If True, use Ollama instead of OpenAI
        use_huggingface: If True, use HuggingFace local models
        ollama_base_url: Ollama server URL (default: http://localhost:11434)
        device: Device for HuggingFace models ("cuda", "cpu", or "auto")
    
    Returns:
        Configured LLM instance (OpenAI, Ollama, or HuggingFace)
    """
    if use_huggingface:
        # Use HuggingFace for local GPU models
        if model in HUGGINGFACE_MODELS:
            model_config = HUGGINGFACE_MODELS[model]
            model_name = model_config["model_name"]
            context_window = model_config["context_window"]
            max_new_tokens = model_config["max_new_tokens"]
            print(f"[HuggingFace] Loading model: {model_name}")
            print(f"   {model_config['description']}")
        else:
            # Custom HuggingFace model
            model_name = model
            context_window = 4096
            max_new_tokens = 512
            print(f"[Warning] Using custom HuggingFace model: {model_name}")
        
        # Detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[HuggingFace] Using device: {device}")
        
        if device == "cuda":
            print(f"[HuggingFace] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[HuggingFace] CUDA Version: {torch.version.cuda}")
        
        # Create HuggingFace LLM with proper configuration
        llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            generate_kwargs={
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
            },
            device_map=device,
            # Use 4-bit quantization to save GPU memory
            model_kwargs={
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "load_in_4bit": True if device == "cuda" else False,
            }
        )
        
        print(f"[OK] HuggingFace LLM configured: {model_name} (temp: {temperature}, device: {device})")
        return llm
        
    elif use_ollama:
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


def setup_embedding_model(
    model: str = "text-embedding-3-small",
    use_huggingface: bool = False,
    device: str = "auto"
) -> OpenAIEmbedding:
    """Setup OpenAI embedding model for queries."""
    if use_huggingface:
        print(f"[HuggingFace] Setting up embedding model: {model}")
        if model in HUGGINGFACE_EMBEDDING_MODELS:
            model_config = HUGGINGFACE_EMBEDDING_MODELS[model]
            model_name = model_config["model_name"]
            embed_dim = model_config["embed_dim"]
            max_length = model_config["max_length"]
            print(f"[HuggingFace] Using embedding model: {model_name}")
        else:
            model_name = model
            embed_dim = 768
            max_length = 512
            print(f"[Warning] Using custom HuggingFace embedding model: {model_name}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=max_length,
            device=device
        )
        
        print(f"Embedding model configured: {model_name} (dim: {embed_dim}, max_length: {max_length})")
        return embed_model
    
    else:
        print(f"[OpenAI] Setting up embedding model: {model}")    
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        embed_model = OpenAIEmbedding(
            model=model,
            api_key=api_key
        )
        
        print(f"Embedding model configured: {model}")
        return embed_model


def configure_global_settings(
    llm: Optional[Union[OpenAI, Ollama, HuggingFaceLLM]] = None,
    llm_model: str = "gpt-4o-mini",
    use_ollama: bool = False,
    use_huggingface: bool = False,
    embed_model: Optional[Union[OpenAIEmbedding, HuggingFaceEmbedding]] = None,
    embedding_model_name: str = "text-embedding-3-small",
    use_huggingface_embeddings: bool = False
):
    """
    Configure LlamaIndex global settings with specific model.
    
    Args:
        llm: Pre-loaded LLM instance. If provided, uses this instead of creating a new one.
        llm_model: Model name to use
        use_ollama: Whether to use Ollama
        use_huggingface: Whether to use HuggingFace
        embed_model: Pre-loaded embedding model. If provided, uses this instead of creating a new one.
        embedding_model_name: Embedding model name
        use_huggingface_embeddings: Whether to use HuggingFace embeddings
    """
    print(f"[configure_global_settings] Configuring with model={llm_model}, ollama={use_ollama}, hf={use_huggingface}")
    print(f"[configure_global_settings] Configuring embeddings with model={embedding_model_name}, hf_embeddings={use_huggingface_embeddings}")
    if llm is not None:
        print("Using provided LLM instance for global settings")
        Settings.llm = llm
    else:
        print("Creating new LLM instance for global settings")
        Settings.llm = setup_llm(
            model=llm_model,
            use_ollama=use_ollama,
            use_huggingface=use_huggingface
        )

    if embed_model is not None:
        print("Using provided embedding model instance for global settings")
        Settings.embed_model = embed_model
    else:
        print("Creating new embedding model instance for global settings")
        Settings.embed_model = setup_embedding_model(
            model=embedding_model_name,
            use_huggingface=use_huggingface_embeddings
        )

    print("Global LlamaIndex settings configured")
