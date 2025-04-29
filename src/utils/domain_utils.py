#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.60
Usage:  Imported by other scripts.
        `from src.utils.domain_utils import load_map, get_domain`
        `domain = get_domain(snomed_code)`

Objective of the Code:
------------
Provides utility functions related to mapping medical concepts (primarily 
SNOMED CT codes) to predefined medical domains (e.g., cardio, metabolic). 
Loads a domain mapping configuration file (YAML or ICD-10 mapping file) and offers a function 
(`get_domain`) to determine the appropriate domain for a given code, 
defaulting to 'general' if no specific mapping is found.
"""

import yaml
import pathlib
import functools
from typing import Dict

# Define path relative to the project root (assuming this file is src/utils/domain_utils.py)
ROOT = pathlib.Path(__file__).resolve().parents[2]
MAP_FILE = ROOT / "configs" / "domain_map.yaml"

@functools.lru_cache(maxsize=None)
def load_map() -> Dict[str, str]:
    """Loads the domain map YAML and converts it to a code -> domain dictionary."""
    if not MAP_FILE.exists():
        # Handle case where map file doesn't exist (e.g., return empty dict or raise error)
        # For now, returning empty dict and letting get_domain default to "general"
        print(f"Warning: Domain map file not found at {MAP_FILE}")
        return {}

    try:
        with MAP_FILE.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading or parsing domain map file {MAP_FILE}: {e}")
        return {}

    code2domain = {}
    if data:
        for dom, dom_data in data.items():
            # Ensure 'codes' key exists and is a list
            if isinstance(dom_data, dict) and 'codes' in dom_data and isinstance(dom_data['codes'], list):
                for c in dom_data['codes']:
                    code2domain[str(c)] = dom
            else:
                print(f"Warning: Skipping domain '{dom}' due to unexpected format in {MAP_FILE}")

    return code2domain

def get_domain(code: str) -> str:
    """Gets the domain for a given code, defaulting to 'general'."""
    # Ensure input is treated as string
    code_str = str(code)
    domain_map = load_map()
    return domain_map.get(code_str, "general")

# Optional: Add a simple test block
if __name__ == "__main__":
    # Example codes (replace with actual codes from your map if possible)
    test_codes = {
        "25000": "metabolic",  # Example, replace with real codes
        "41400": "cardio",     # Example, replace with real codes
        "12345": "general"     # Example code not in map
    }

    print(f"Loading map from: {MAP_FILE}")
    # Trigger loading the map
    load_map()

    print("\nTesting get_domain:")
    for code, expected in test_codes.items():
        actual = get_domain(code)
        print(f"  Code: {code}, Expected: {expected}, Actual: {actual} -> {'OK' if actual == expected else 'FAIL'}")

    # Test cache (optional)
    print("\nCache Info:")
    print(load_map.cache_info())