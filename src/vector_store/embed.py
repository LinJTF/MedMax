"""OpenAI embedding generation."""

import os
import time
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (1 token ≈ 4 characters)."""
    return len(text) // 4


def process_embedding_batch(client: OpenAI, batch: List[str], model: str, batch_num: int) -> List[List[float]]:
    """Process a single batch of texts for embedding."""
    print(f"Processing batch {batch_num} with {len(batch)} texts")
    
    try:
        response = client.embeddings.create(input=batch, model=model)
        embeddings = [embedding_data.embedding for embedding_data in response.data]
        print(f"✅ Batch {batch_num} completed successfully")
        time.sleep(0.5)  # Rate limiting
        return embeddings
        
    except Exception as e:
        print(f"❌ Error processing batch {batch_num}: {e}")
        return process_individual_texts(client, batch, model)


def process_individual_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """Process texts individually when batch fails."""
    print("Retrying with individual texts...")
    embeddings = []
    
    for text in texts:
        try:
            response = client.embeddings.create(input=[text], model=model)
            embeddings.append(response.data[0].embedding)
            time.sleep(0.2)
        except Exception as e:
            print(f"❌ Failed to process individual text: {e}")
            embeddings.append([0.0] * 1536)  # Fallback zero vector
    
    return embeddings


def create_smart_batches(texts: List[str], max_tokens: int = 250000) -> List[List[str]]:
    """Create batches based on token estimation."""
    batches = []
    current_batch = []
    current_tokens = 0
    total_tokens = 0
    
    print("📦 Creating smart batches based on token estimation...")
    
    for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
        text_tokens = estimate_tokens(text)
        total_tokens += text_tokens
        
        # Start new batch if current would exceed limit
        if current_tokens + text_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            print(f"  📦 Batch {len(batches)} created: {len(current_batch)} texts, ~{current_tokens:,} tokens")
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    # Add final batch
    if current_batch:
        batches.append(current_batch)
        print(f"  📦 Final batch {len(batches)} created: {len(current_batch)} texts, ~{current_tokens:,} tokens")
    
    print(f"📊 Batching complete:")
    print(f"  - Total estimated tokens: {total_tokens:,}")
    print(f"  - Total batches created: {len(batches)}")
    print(f"  - Average tokens per batch: {total_tokens // len(batches) if batches else 0:,}")
    
    return batches


def generate_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generate embeddings using OpenAI API with smart batching."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=api_key)
    
    print(f"🚀 Processing {len(texts)} texts for embedding generation...")
    
    # Create smart batches
    batches = create_smart_batches(texts)
    print(f"📋 Created {len(batches)} batches for processing")
    
    # Process each batch with progress bar
    all_embeddings = []
    
    with tqdm(total=len(batches), desc="🤖 Processing batches", unit="batch") as pbar:
        for batch_num, batch in enumerate(batches, 1):
            batch_embeddings = process_embedding_batch(client, batch, model, batch_num)
            all_embeddings.extend(batch_embeddings)
            
            # Update progress bar with detailed info
            pbar.update(1)
            pbar.set_postfix({
                'embeddings': len(all_embeddings),
                'batch_size': len(batch),
                'total_texts': len(texts)
            })
    
    print(f"✅ Generated {len(all_embeddings)} embeddings total")
    return all_embeddings
