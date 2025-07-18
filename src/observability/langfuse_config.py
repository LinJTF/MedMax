import os
from langfuse import Langfuse, observe
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
