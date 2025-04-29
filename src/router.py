#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.25
Usage:  Used via import by other scripts (e.g., evaluate.py, quick_demo.py).
        `router = Router(config_path='domain_map.yaml')`
        `selected_agents = router.predict(prompt)`

Objective of the Code:
------------
Implements the Layer 1 prompt router for the MoA pipeline. It takes a patient 
prompt as input and uses a configuration file (domain_map.yaml) containing 
keywords mapped to specialist agent IDs. The router identifies relevant keywords 
in the prompt (case-insensitive) and returns a list of agent IDs designated to 
handle that prompt.
"""
import os
import logging
import re
import pathlib
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional, Pattern

# Import domain utility functions (Updated)
try:
    from utils.domain_utils import get_domain
except ImportError:
    # Handle case when imported from parent directory
    from src.utils.domain_utils import get_domain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KeywordRouter")

# --- Constants ---
# Use environment variable for base model path
DEFAULT_BASE_MODEL_PATH = os.environ.get("DEFAULT_PHI2_BASE_PATH", "path/to/your/phi-2/model") # Placeholder if env var not set
MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
AGENTS_DIR = MODELS_DIR / "agents"

class Router:
    """
    Routes a text prompt to appropriate agent IDs based on keyword matching
    defined in a YAML configuration file.
    """

    def __init__(self, config_path: str | pathlib.Path):
        """
        Initialize the router with the path to the domain map configuration file.

        Args:
            config_path: Path to the domain map YAML file.
                         Expected format:
                         domain_name:
                           keywords: [list, of, keywords]
                           agent_id: agent_folder_name_in_checkpoints
        """
        self.config_path = pathlib.Path(config_path)
        self.domain_map: Dict[str, Dict[str, str | List[str]]] = {}
        self.compiled_patterns: Dict[str, Pattern] = {}
        self.agent_id_map: Dict[str, str] = {} # domain -> agent_id

        if not self.config_path.exists():
            logger.error(f"Router config file not found: {self.config_path}")
            raise FileNotFoundError(f"Router config file not found: {self.config_path}")

        self._load_config()
        self._compile_patterns()
        logger.info(f"Router initialized with {len(self.domain_map)} domains from {self.config_path}")
        logger.debug(f"Agent ID mapping: {self.agent_id_map}")

    def _load_config(self):
        """Loads the domain mapping from the YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                if not isinstance(loaded_config, dict):
                    raise ValueError("Domain map YAML must load as a dictionary.")
                self.domain_map = loaded_config
                # Populate agent_id map
                for domain, data in self.domain_map.items():
                    if isinstance(data, dict) and 'agent_id' in data:
                        self.agent_id_map[domain] = data['agent_id']
                    else:
                        logger.warning(f"Missing 'agent_id' for domain '{domain}' in {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {self.config_path}: {e}")
            raise ValueError(f"Invalid YAML format in {self.config_path}") from e
        except Exception as e:
            logger.error(f"Error loading config file {self.config_path}: {e}")
            raise

    def _compile_patterns(self):
        """Compiles regex patterns for keyword matching for each domain."""
        for domain, data in self.domain_map.items():
            if not isinstance(data, dict) or 'keywords' not in data or not isinstance(data['keywords'], list):
                logger.warning(f"Skipping domain '{domain}': Missing or invalid 'keywords' list in {self.config_path}")
                continue
                
            keywords = data['keywords']
            # Create a regex pattern that matches any of the keywords, case-insensitive
            # Ensure keywords are escaped for regex safety and use word boundaries
            escaped_keywords = [re.escape(kw) for kw in keywords if kw] # Filter empty strings
            if not escaped_keywords:
                 logger.warning(f"No valid keywords found for domain '{domain}'.")
                 continue
                 
            pattern_str = r"\b(" + "|".join(escaped_keywords) + r")\b"
            try:
                self.compiled_patterns[domain] = re.compile(pattern_str, re.IGNORECASE)
                logger.debug(f"Compiled pattern for domain '{domain}': {pattern_str}")
            except re.error as e:
                logger.error(f"Invalid regex pattern for domain '{domain}': {pattern_str}. Error: {e}")
                # Optionally skip this domain or raise an error
                
    def predict(self, prompt: str) -> List[str]:
        """
        Predicts which agent(s) should handle the prompt based on keyword matching.

        Args:
            prompt: The input text prompt.

        Returns:
            A list of unique agent_id strings corresponding to matched domains.
            Returns an empty list if no keywords are matched.
        """
        matched_agent_ids = set()
        
        if not prompt:
            logger.warning("Received empty prompt. Cannot perform routing.")
            return []

        logger.debug(f"Routing prompt: {prompt[:150]}...")
        for domain, pattern in self.compiled_patterns.items():
            if pattern.search(prompt):
                agent_id = self.agent_id_map.get(domain)
                if agent_id:
                    logger.info(f"Keyword match found for domain '{domain}'. Adding agent: '{agent_id}'")
                    matched_agent_ids.add(agent_id)
                else:
                     logger.warning(f"Keyword match for domain '{domain}', but no corresponding agent_id found in map.")

        if not matched_agent_ids:
            logger.info("No matching domain keywords found in prompt.")
            # Decide fallback behavior: empty list, or default agent?
            # Returning empty list - caller (e.g., evaluate.py) should handle fallback.

        return sorted(list(matched_agent_ids)) # Return sorted list for consistency

# Example usage
if __name__ == "__main__":
    # Assume domain_map.yaml exists in the project root
    # Use a relative path from the script location
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent
    default_config_path = project_root / "domain_map.yaml"

    if not default_config_path.exists():
         print(f"Error: Default domain_map.yaml not found at {default_config_path}")
         print("Cannot run demo.")
         exit(1)

    print(f"Initializing Router with config: {default_config_path}")
    router = Router(config_path=default_config_path)

    # Example prompts
    prompt_cardio = "Patient presents with chest pain and palpitations after exertion."
    prompt_gi = "45-year-old female reports persistent heartburn and abdominal pain, worse after eating."
    prompt_musculo = "Fell yesterday, complaining of severe ankle pain and swelling."
    prompt_mixed = "Elderly patient with shortness of breath, history of heart failure and recent knee injury."
    prompt_none = "Routine physical exam for a healthy 20-year-old."

    print("\n--- Demo Prompts --- ")
    
    print(f"\nPrompt (Cardio): {prompt_cardio}")
    agents_cardio = router.predict(prompt_cardio)
    print(f"  >> Selected Agents: {agents_cardio}")

    print(f"\nPrompt (GI): {prompt_gi}")
    agents_gi = router.predict(prompt_gi)
    print(f"  >> Selected Agents: {agents_gi}")

    print(f"\nPrompt (Musculo): {prompt_musculo}")
    agents_musculo = router.predict(prompt_musculo)
    print(f"  >> Selected Agents: {agents_musculo}")
    
    print(f"\nPrompt (Mixed): {prompt_mixed}")
    agents_mixed = router.predict(prompt_mixed)
    print(f"  >> Selected Agents: {agents_mixed}")
    
    print(f"\nPrompt (None): {prompt_none}")
    agents_none = router.predict(prompt_none)
    print(f"  >> Selected Agents: {agents_none}") 