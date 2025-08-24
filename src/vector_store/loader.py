import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm


def load_pubmedqa_parquet_data(
    data_dir: str = "data",
    limit_per_dataset: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load PubMedQA data from parquet files with deduplication.
    
    Args:
        data_dir: Directory containing the parquet files
        limit_per_dataset: Optional limit for number of records per dataset
        
    Returns:
        Tuple of (unique_records, statistics)
    """
    print("Loading PubMedQA datasets from parquet files...")
    
    data_path = Path(data_dir)
    unlabeled_path = data_path / "pqa_unlabeled_train.parquet"
    labeled_path = data_path / "pqa_labeled_train.parquet"
    artificial_path = data_path / "pqa_artificial_train.parquet"
    
    missing_files = []
    for path in [unlabeled_path, labeled_path, artificial_path]:
        if not path.exists():
            missing_files.append(str(path))
    
    if missing_files:
        print(f"Missing parquet files: {missing_files}")
        return [], {"error": "missing_files"}
    

    df_unlabeled = pd.read_parquet(unlabeled_path)
    df_labeled = pd.read_parquet(labeled_path)
    df_artificial = pd.read_parquet(artificial_path)
    
    print("Dataset sizes:")
    print(f"  - Unlabeled: {len(df_unlabeled)} records")
    print(f"  - Labeled: {len(df_labeled)} records") 
    print(f"  - Artificial: {len(df_artificial)} records")
    
    if limit_per_dataset:
        print(f"Applying limit of {limit_per_dataset} records per dataset")
        df_unlabeled = df_unlabeled.head(limit_per_dataset)
        df_labeled = df_labeled.head(limit_per_dataset)
        df_artificial = df_artificial.head(limit_per_dataset)
    
    unique_records = {}
    stats = {"unlabeled": 0, "labeled": 0, "artificial": 0, "duplicates": 0}
    
    def process_dataset(df: pd.DataFrame, dataset_name: str, has_final_decision: bool = False):
        """Process a single dataset and extract unique records."""
        dataset_stats = 0
        
        for idx, row in tqdm(df.iterrows(), desc=f"Processing {dataset_name}", total=len(df)):
            pubid = row['pubid']
            
            # Skip if we already have this pubid (deduplication)
            if pubid in unique_records:
                stats["duplicates"] += 1
                continue
            
            contexts_list = row['context']['contexts'].tolist()
            
            record_data = {
                'question': row['question'],
                'contexts': contexts_list,
                'long_answer': row['long_answer'],
                'pubid': pubid,
                'dataset_source': dataset_name
            }
            
            if has_final_decision and 'final_decision' in row:
                record_data['final_decision'] = row['final_decision']
            
            unique_records[pubid] = record_data
            dataset_stats += 1
        
        return dataset_stats
    
    stats["unlabeled"] = process_dataset(df_unlabeled, "unlabeled", False)
    stats["labeled"] = process_dataset(df_labeled, "labeled", True) 
    stats["artificial"] = process_dataset(df_artificial, "artificial", True)
    
    unique_records_list = list(unique_records.values())
    
    print("Extraction Results:")
    print(f"  - Unique records from unlabeled: {stats['unlabeled']}")
    print(f"  - Unique records from labeled: {stats['labeled']}")
    print(f"  - Unique records from artificial: {stats['artificial']}")
    print(f"  - Total unique records: {len(unique_records_list)}")
    print(f"  - Duplicates avoided: {stats['duplicates']}")
    
    return unique_records_list, stats


def load_pubmed_data(jsonl_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load PubMed data from JSONL file with optional limit."""
    pubmed_data = []
    
    print(f"Loading PubMed data from {jsonl_path}")
    if limit:
        print(f"Limiting to {limit} records for testing")

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        print("Counting total lines...")
        total_lines = sum(1 for _ in file)
        file.seek(0)
        
        max_records = min(total_lines, limit) if limit else total_lines
        progress_bar = tqdm(total=max_records, desc="Loading records")
        
        for line_num, line in enumerate(file, 1):
            if limit and len(pubmed_data) >= limit:
                print(f"Reached limit of {limit} records, stopping...")
                break
                
            try:
                data = json.loads(line.strip())
                if all(key in data.get('metadata', {}) for key in ['question', 'final_decision']) and \
                   'content' in data and 'contexts' in data['content'] and \
                   'long_answer' in data['content']:
                    
                    extracted_data = {
                        'question': data['metadata']['question'],
                        'contexts': data['content']['contexts'],
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
    """Format PubMed records for embedding generation - using ONLY context text."""
    formatted_texts = []
    truncated_final = 0
    
    print("Formatting context texts for embedding...")
    
    for record in tqdm(pubmed_records, desc="Formatting records"):
        # Use ONLY the contexts for embedding (RAG retrieval)
        contexts_text = " ".join(record['contexts'])
        
        # Truncate very long contexts to prevent token limit issues
        if len(contexts_text) > max_length:
            contexts_text = contexts_text[:max_length] + "..."
            truncated_final += 1
        
        # Use only context text for embedding
        formatted_text = contexts_text
        
        formatted_texts.append(formatted_text)
    
    print("Text formatting statistics:")
    print(f"  - Truncated contexts: {truncated_final}")
    print(f"  - Average text length: {sum(len(t) for t in formatted_texts) // len(formatted_texts)} chars")
    print(f"  - Max text length: {max(len(t) for t in formatted_texts)} chars")
    
    return formatted_texts
