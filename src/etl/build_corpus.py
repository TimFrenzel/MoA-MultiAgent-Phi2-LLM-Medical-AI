#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 3.10
Usage: python -m src.etl.build_corpus [--jsonl_path <path>] [--arrow_dir <path>] [--limit N]

Objective of the Code:
------------
Generates (if needed) and processes patient prompt data to create domain-specific
training and testing datasets for fine-tuning specialist medical agents.
It reads prompts (from JSONL or generated via `src.prompt_gen`), maps samples
to domains using `src.utils.domain_utils`, splits them, and saves the resulting
datasets in Arrow format under the specified output directory (e.g., `data/arrow/<domain>`).
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Import domain utility functions
# Assuming script is run as a module (python -m src.etl.build_corpus)
from ..utils.domain_utils import load_map, get_domain
from ..prompt_gen import stream_rows, count_rows

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
# Use pathlib for more robust path handling
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] # Go two levels up (src/etl -> src -> project root)
DEFAULT_JSONL_PATH = PROJECT_ROOT / "data" / "prompts.jsonl"
ARROW_DIR = PROJECT_ROOT / "data" / "arrow"

def load_jsonl_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file into a list of dictionaries.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    logger.info(f"Loading data from {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found: {jsonl_path}")
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                data.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing JSON line: {e}")
                continue
                
    logger.info(f"Loaded {len(data)} samples from JSONL file")
    return data

def filter_dataset_by_domain(dataset: Dataset, domain: str) -> Dataset:
    """
    Filter a dataset to include only samples matching a specific domain.
    
    Args:
        dataset: Dataset to filter
        domain: Domain to filter by (e.g., 'cardio', 'neuro')
        
    Returns:
        Filtered dataset
    """
    logger.info(f"Filtering dataset for domain: {domain}")
    
    # Define a filter function to check if a sample belongs to this domain
    def filter_fn(example):
        # Extract the target SNOMED CT code from the sample
        # The prompt_gen script stores this in the 'label' field
        snomed_code = example.get("label")

        # If no code found, skip this sample
        if snomed_code is None:
            return False

        # Check if the code's domain matches the target domain
        # Ensure code is treated as string for lookup in domain_utils
        return get_domain(str(snomed_code)) == domain

    # Apply the filter
    filtered_dataset = dataset.filter(filter_fn)
    logger.info(f"Found {len(filtered_dataset)} samples for domain '{domain}'")
    
    return filtered_dataset

def create_train_test_split(dataset: Dataset, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Split a dataset into training and testing sets.
    
    Args:
        dataset: Dataset to split
        test_size: Proportion of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict containing 'train' and 'test' splits
    """
    logger.info(f"Creating train/test split with test_size={test_size}")
    
    # Shuffle and split the dataset
    shuffled = dataset.shuffle(seed=seed)
    train_size = int(len(shuffled) * (1 - test_size))
    
    train_dataset = shuffled.select(range(train_size))
    test_dataset = shuffled.select(range(train_size, len(shuffled)))
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset_dict

def save_domain_datasets(datasets: Dict[str, DatasetDict], output_dir: str) -> None:
    """
    Save domain-specific datasets to disk in Arrow format.
    
    Args:
        datasets: Dictionary of domain-specific DatasetDicts
        output_dir: Directory to save datasets
    """
    logger.info(f"Saving domain datasets to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each domain dataset
    for domain, dataset_dict in datasets.items():
        domain_dir = os.path.join(output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        # Save train and test splits
        for split, ds in dataset_dict.items():
            split_dir = os.path.join(domain_dir, split)
            ds.save_to_disk(split_dir)
            
        logger.info(f"Saved {domain} dataset ({len(dataset_dict['train'])} train, {len(dataset_dict['test'])} test) to {domain_dir}")

def generate_prompts_if_needed(jsonl_path: str, limit: Optional[int] = None) -> None:
    """Generate prompt data if it doesn't exist, applying an optional limit."""
    if os.path.exists(jsonl_path):
        logger.info(f"Prompt data already exists at {jsonl_path}")
        return

    logger.info("Prompt data not found, generating prompts from database...")
    try:
        # Get total *potential* count first (allow_repeats=True matches the loop below)
        logger.info("Getting total potential prompt count from database...")
        full_count = count_rows(allow_repeats=True)
        logger.info(f"Database could potentially yield {full_count:,} prompts.")

        # Determine the actual number of prompts to generate
        generate_limit = None
        tqdm_total = full_count # Default total for tqdm
        if limit is not None and limit > 0:
            if limit >= full_count:
                logger.info(f"Requested limit ({limit:,}) is >= potential count ({full_count:,}). Generating all prompts.")
                # generate_limit remains None
            else:
                logger.info(f"Applying limit: generating {limit:,} prompts out of {full_count:,} potential.")
                generate_limit = limit # Apply limit in DB query
                tqdm_total = limit # Adjust tqdm total to the limit
        else:
            logger.info("No limit applied. Generating all potential prompts.")
            # generate_limit remains None

        # Stream samples from the database and save to JSONL
        logger.info(f"Saving generated prompts to {jsonl_path}...")
        # Ensure the target directory exists
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

        # Pass generate_limit to stream_rows and tqdm_total to tqdm
        tqdm_iterator = tqdm(
            stream_rows(allow_repeats=True, limit=generate_limit), # Pass limit here
            total=tqdm_total, # Use calculated total for progress bar
            desc="Generating prompts",
            unit="prompt"
        )
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            # Force allow_repeats=True used in count_rows and stream_rows
            for sample in tqdm_iterator:
                f.write(json.dumps(sample) + '\n')

        # Log the final count obtained from tqdm
        final_count = tqdm_iterator.n
        logger.info(f"Finished generating and saved {final_count:,} prompt samples to {jsonl_path}")

    except Exception as e:
        logger.error(f"Failed to generate prompts: {e}")
        raise

def build_corpus(jsonl_path: str, arrow_dir: str, skip_generate: bool = False, limit: Optional[int] = None) -> None:
    """Build corpus for medical diagnosis agent training."""
    # Generate prompts if needed, passing the limit
    if not skip_generate:
        generate_prompts_if_needed(jsonl_path, limit=limit) # Pass limit
    elif not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found at {jsonl_path} and skip_generate=True")
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    # Load data from JSONL file
    data = load_jsonl_data(jsonl_path)

    # Create a dataset from the data
    dataset = Dataset.from_list(data)
    logger.info(f"Created dataset with {len(dataset)} samples")

    # Get available domains from the loaded map
    # Load the map once
    code_to_domain_map = load_map()
    # Extract unique domain names (including potentially 'general' if it's in the map explicitly)
    # Add 'general' explicitly if not present, as it's handled separately
    domains_from_map = set(code_to_domain_map.values())
    available_domains = list(domains_from_map) # Convert set to list
    if 'general' not in available_domains:
         available_domains.append('general') # Ensure general is always considered
    # Sort for consistent processing order (optional)
    available_domains.sort()

    logger.info(f"Processing {len(available_domains)} domains: {', '.join(available_domains)}")

    # Create domain-specific datasets
    domain_datasets = {}
    domain_stats = {}

    # Handle the 'general' domain dataset first (all samples)
    if 'general' in available_domains:
        logger.info("Processing general domain (all samples)...")
        general_dataset = create_train_test_split(dataset)
        domain_datasets['general'] = general_dataset
        domain_stats['general'] = {
            'train': len(general_dataset['train']),
            'test': len(general_dataset['test']),
            'total': len(general_dataset['train']) + len(general_dataset['test'])
        }

    # Create domain-specific datasets for other domains
    for domain in available_domains:
        if domain == 'general':
            continue  # Already handled

        # Filter dataset by domain
        domain_dataset = filter_dataset_by_domain(dataset, domain)

        # Only create split if we have enough samples
        if len(domain_dataset) > 10:
            domain_split = create_train_test_split(domain_dataset)
            domain_datasets[domain] = domain_split
            domain_stats[domain] = {
                'train': len(domain_split['train']),
                'test': len(domain_split['test']),
                'total': len(domain_split['train']) + len(domain_split['test'])
            }
        else:
            logger.warning(f"Not enough samples for domain '{domain}': {len(domain_dataset)} < 10")
            domain_stats[domain] = {
                'train': 0,
                'test': 0,
                'total': len(domain_dataset)
            }

    # Save domain datasets
    save_domain_datasets(domain_datasets, arrow_dir)

    # Print domain statistics
    print("\nDomain Statistics:")
    print("=" * 60)
    print(f"{'Domain':<15} {'Total':<10} {'Train':<10} {'Test':<10}")
    print("-" * 60)
    for domain, stats in domain_stats.items():
        print(f"{domain:<15} {stats['total']:<10} {stats['train']:<10} {stats['test']:<10}")
    print("=" * 60)
    
    logger.info(f"Successfully created domain-specific datasets in {arrow_dir}")
    print(f"\nDomain-specific datasets created successfully in {arrow_dir}")
    print("You can now train specialized medical agents using:")
    print(f"  python src/train_agent.py --domain [DOMAIN] --model_path models/agents/[DOMAIN]")

def main():
    """Main function to parse arguments and build corpus."""
    parser = argparse.ArgumentParser(description="Build corpus for medical diagnosis agent training")
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL_PATH,
                        help="Path to the JSONL file containing prompt data")
    parser.add_argument("--arrow_dir", type=str, default=ARROW_DIR,
                        help="Directory to save Arrow format datasets")
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip prompt generation and use existing JSONL file")
    parser.add_argument("--config", type=str, default="configs/domain_map.yaml",
                        help="Path to domain configuration file")
    # Add the limit argument
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of prompts to generate (processes all existing prompts if --skip_generate)")

    args = parser.parse_args()

    try:
        # Pass the limit argument to build_corpus
        build_corpus(args.jsonl_path, args.arrow_dir, args.skip_generate, limit=args.limit)
    except Exception as e:
        logger.error(f"Error building corpus: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
