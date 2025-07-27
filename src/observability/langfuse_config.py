import os
from langfuse import Langfuse, observe, get_client
from dotenv import load_dotenv
from typing import Any

load_dotenv()

# Simple global client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

def create_session(name: str) -> str:
    """Create session and return session ID"""
    session = langfuse.create_session(name=name)
    return session.id

def update_trace_metadata(metadata: dict[str, Any]) -> None:
    """Update current trace metadata"""
    try:
        client = get_client()
        client.update_current_trace(metadata=metadata)
    except Exception as e:
        print(f"Error updating trace metadata: {e}")
        
def update_span_metadata(metadata: dict[str, Any]) -> None:
    """Update current span metadata"""
    try:
        client = get_client()
        client.update_current_span(metadata=metadata)
    except Exception as e:
        print(f"Error updating span metadata: {e}")

def update_current_generation(
    model: str,
    input_text: str,
    output_text: str,
    input_tokens: int,
    output_tokens: int,
    input_cost: float,
    output_cost: float,
    metadata: dict[str, Any] | None = None
) -> None:
    """
    Update current generation with LLM-specific data using Langfuse format.
    
    Args:
        model: Model name (e.g., 'gpt-4o-mini')
        input_text: The prompt/input text
        output_text: The model's response
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens  
        input_cost: Cost for input tokens in USD
        output_cost: Cost for output tokens in USD
        metadata: Additional metadata (optional)
    """
    try:
        client = get_client()
        
        update_data = {
            "model": model,
            "input": input_text,
            "output": output_text,
            "usage_details": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "cost_details": {
                "input": input_cost,
                "output": output_cost,
                "total": input_cost + output_cost
            }
        }

        if metadata:
            update_data["metadata"] = metadata
        
        client.update_current_generation(**update_data)
        
    except Exception as e:
        print(f"Error updating current generation: {e}")
