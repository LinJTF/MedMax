import tiktoken
from transformers import AutoTokenizer
from functools import lru_cache

OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006    
    },
    "gpt-4o": {
        "input": 0.0025,      
        "output": 0.01      
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,    
        "output": 0.0015    
    }
}

# Mapping of Ollama models to HuggingFace model IDs for tokenizers
OLLAMA_TO_HF_TOKENIZER = {
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    # phi3:mini removed - requires 50GB RAM for 128K context
}

@lru_cache(maxsize=10)
def _get_ollama_tokenizer(model_name: str):
    """Cache tokenizers to avoid reloading."""
    hf_model = OLLAMA_TO_HF_TOKENIZER.get(model_name)
    if hf_model:
        return AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    return None

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in a text string for a given model."""
    
    # For Ollama models, use proper tokenizers
    if ":" in model or model.startswith(("mistral", "phi", "qwen", "gemma", "llama")):
        try:
            tokenizer = _get_ollama_tokenizer(model)
            if tokenizer:
                return len(tokenizer.encode(text, add_special_tokens=False))
            else:
                # Fallback to approximation if tokenizer not found
                # Based on typical tokenizer behavior: ~1.3 tokens per word
                return int(len(text.split()) * 1.3)
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model}, using approximation: {e}")
            return int(len(text.split()) * 1.3)
    
    # For OpenAI models, use tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except KeyError:
        # Fallback to cl100k_base (GPT-4 encoding) if model not recognized
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> dict[str, float]:
    """Calculate the cost of input and output tokens for a given model."""
    # Ollama models are free (local)
    if ":" in model or model.startswith(("mistral", "phi", "qwen", "gemma", "llama")):
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0
        }
    
    # OpenAI models have costs
    if model not in OPENAI_PRICING:
        # Default to gpt-4o-mini pricing if model not recognized
        pricing = OPENAI_PRICING["gpt-4o-mini"]
    else:
        pricing = OPENAI_PRICING[model]
    
    input_cost = (input_tokens/1000) * pricing["input"]
    output_cost = (output_tokens/1000) * pricing["output"]
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

def get_token_usage(prompt: str, response: str, model: str = "gpt-4o-mini") -> dict[str, int]:
    """Get token usage and cost for a given prompt and response."""
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
