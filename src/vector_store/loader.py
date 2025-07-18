"""Data loading and formatting for PubMed data."""

import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm


def load_pubmed_data(jsonl_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load PubMed data from JSONL file with optional limit."""
    pubmed_data = []
    
    print(f"Loading PubMed data from {jsonl_path}")
    if limit:
        print(f"🔢 Limiting to {limit} records for testing")
    
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        # Count total lines for progress bar
        print("Counting total lines...")
        total_lines = sum(1 for _ in file)
        file.seek(0)
        
        max_records = min(total_lines, limit) if limit else total_lines
        progress_bar = tqdm(total=max_records, desc="Loading records")
        
        for line_num, line in enumerate(file, 1):
            # Stop if we reached the limit
            if limit and len(pubmed_data) >= limit:
                print(f"Reached limit of {limit} records, stopping...")
                break
                
            try:
                data = json.loads(line.strip())
                # Extract only the fields we need
                if all(key in data.get('metadata', {}) for key in ['question', 'final_decision']) and \
                   'content' in data and 'contexts' in data['content'] and \
                   'long_answer' in data['content']:
                    
                    extracted_data = {
                        'question': data['metadata']['question'],
                        'contexts': data['content']['contexts'],  # This is the list of contexts
                        'final_decision': data['metadata']['final_decision'],
                        'long_answer': data['content']['long_answer']
                    }
                    pubmed_data.append(extracted_data)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue
            
            progress_bar.update(1)
            
            # Update progress bar description with current count
            if len(pubmed_data) > 0 and len(pubmed_data) % 100 == 0:
                progress_bar.set_description(f"Loading records (found {len(pubmed_data)} valid)")
        
        progress_bar.close()
    
    print(f"Loaded {len(pubmed_data)} valid PubMed records")
    return pubmed_data


def format_pubmed_for_embedding(
    pubmed_records: List[Dict[str, Any]], 
    max_length: int = 7000
) -> List[str]:
    """Format PubMed records for embedding generation with length limits."""
    formatted_texts = []
    truncated_contexts = 0
    truncated_answers = 0
    truncated_final = 0
    
    print("Formatting texts for embedding...")
    
    for record in tqdm(pubmed_records, desc="Formatting records"):
        # Combine contexts into a single text
        contexts_text = " ".join(record['contexts'])
        
        # Truncate very long contexts to prevent token limit issues
        if len(contexts_text) > 4000:
            contexts_text = contexts_text[:4000] + "..."
            truncated_contexts += 1
        
        # Truncate very long answers
        long_answer = record['long_answer']
        if len(long_answer) > 2000:
            long_answer = long_answer[:2000] + "..."
            truncated_answers += 1
        
        # Create comprehensive text for embedding
        formatted_text = (
            f"Question: {record['question']} "
            f"Context: {contexts_text} "
            f"Answer: {record['final_decision']} "
            f"Explanation: {long_answer}"
        )
        
        # Final length check and truncation
        if len(formatted_text) > max_length:
            formatted_text = formatted_text[:max_length] + "..."
            truncated_final += 1
        
        formatted_texts.append(formatted_text)
    
    # Print statistics
    print("Text formatting statistics:")
    print(f"  - Truncated contexts: {truncated_contexts}")
    print(f"  - Truncated answers: {truncated_answers}")
    print(f"  - Truncated final texts: {truncated_final}")
    print(f"  - Average text length: {sum(len(t) for t in formatted_texts) // len(formatted_texts)} chars")
    print(f"  - Max text length: {max(len(t) for t in formatted_texts)} chars")
    
    return formatted_texts
