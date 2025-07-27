import tiktoken

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

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in a text string for a given model."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def calculate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> dict[str, float]:
    """Calculate the cost of input and output tokens for a given model."""
    if model not in OPENAI_PRICING:
        raise ValueError(f"Model {model} not supported for cost calculation.")
    
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
