#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.1
Usage:  
    python src/train_agent.py --agent cardio
    python src/train_agent.py --agent metabolic
    python src/train_agent.py --agent general --allow_repeats
    python src/train_agent.py --agent cardio --data_dir data/arrow/cardio

Objective:
-----------
Fine-tune Phi-2 using QLoRA technique for the next-diagnosis prediction task.
Creates a specialized agent for a specific medical domain using the prompt datasets
prepared by build_corpus.py.

Updates in v1.1:
- Added support for domain-specific datasets
- Improved data loading logic to use domain-specific directories
"""
import os
import sys
import argparse
import pathlib
import yaml
import json
from datetime import datetime
import logging
import numpy as np
import evaluate

import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import load_from_disk
from trl import SFTTrainer

# --- Paths ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "arrow"
LOGS_DIR = ROOT / "logs"

# --- Constants ---
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Metrics --- 
# Remove the complex compute_metrics function that requires storing all logits
# Function removed: def compute_metrics(eval_pred):...

# --- Functions ---
def setup_logging(agent_name):
    """Set up logging to file and console."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up file handler for this agent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"train_{agent_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logging set up for agent: {agent_name}")
    logger.info(f"Logs will be saved to: {log_file}")

def load_config(config_path):
    """Load the agent configuration from the config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def update_config_with_args(config, args):
    """Update the agent configuration with command line arguments."""
    # Create config if it doesn't exist
    if not config:
        config = {}
    
    # Initialize agents section if it doesn't exist
    if 'agents' not in config:
        config['agents'] = {}
    
    # Get existing agent config or create it
    agent_config = config['agents'].get(args.agent, {})
    
    # Update agent config with command line args
    agent_config.update({
        'agent_name': args.agent,
        'allow_repeats': args.allow_repeats,
        'data_dir': args.data_dir,  # Add data_dir to agent config
    })
    
    # Use default LoRA config if not specified
    if 'lora' not in agent_config:
        agent_config['lora'] = DEFAULT_LORA_CONFIG
    
    # Use command-line overrides if provided
    if args.lora_r:
        agent_config['lora']['r'] = args.lora_r
    if args.lora_alpha:
        agent_config['lora']['lora_alpha'] = args.lora_alpha
    if args.lora_dropout:
        agent_config['lora']['lora_dropout'] = args.lora_dropout
    
    # Set model paths
    agent_config['base_model'] = args.base_model
    agent_config['output_dir'] = str(ROOT / "checkpoints" / f"{args.agent}_lora")
    
    # Update config
    config['agents'][args.agent] = agent_config
    
    return config

def prepare_model_and_tokenizer(model_path):
    """Load the base model and tokenizer with quantization."""
    logger.info(f"Loading base model and tokenizer from: {model_path}")

    # Prepare quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load tokenizer directly from the model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load model with quantization from the same path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def load_dataset(data_dir, allow_repeats):
    """Load the dataset from disk.
    
    Args:
        data_dir: Path to the directory containing the dataset
        allow_repeats: Whether to allow repeating diagnoses in dataset
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset from: {data_dir}")
    
    # Check if data_dir exists
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check for train/test subdirectories
    train_path = data_path / "train"
    test_path = data_path / "test"
    
    if not train_path.exists() or not test_path.exists():
        logger.error(f"Train or test dataset not found in {data_dir}")
        logger.error("Make sure to run build_corpus.py first to create domain-specific datasets")
        logger.error("Example: python src/build_corpus.py --domains cardio,neuro,general")
        sys.exit(1)
    
    # Load train and test datasets
    try:
        train_dataset = load_from_disk(str(train_path))
        eval_dataset = load_from_disk(str(test_path))
        
        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(eval_dataset)} evaluation examples")
        
        return train_dataset, eval_dataset
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

def generate_example_predictions(model, tokenizer, dataset, num_examples=3, max_length=100):
    """Generate example predictions for qualitative analysis.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer for encoding/decoding
        dataset: The dataset to sample from  
        num_examples: Number of examples to generate
        max_length: Maximum length of generated sequences
        
    Returns:
        List of dictionaries containing input and predicted output
    """
    logger.info(f"Generating {num_examples} example predictions for analysis")
    examples = []
    
    # Setup model for generation
    model.eval()
    model.config.use_cache = True  # Enable cache for faster generation
    
    # Sample random examples 
    import random
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    
    for idx in indices:
        example = dataset[idx]
        input_text = example.get("text", "")  # Get input text based on dataset format
        
        # Handle case when "text" field might not exist (adapt to your dataset structure)
        if not input_text and "input_ids" in example:
            input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        
        # Find a reasonable prompt cutoff point - the user's query for instance
        prompt_cutoff = input_text.find("\nAI:") if "\nAI:" in input_text else len(input_text) // 2
        prompt = input_text[:prompt_cutoff].strip()
        expected = input_text[prompt_cutoff:].strip()
        
        # Generate prediction
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated output
            predicted = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Add to examples
            examples.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted
            })
            
            logger.info(f"Example {len(examples)}:")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Expected: {expected}")
            logger.info(f"Predicted: {predicted}")
            logger.info("=" * 40)
            
        except Exception as e:
            logger.warning(f"Error generating prediction for example {idx}: {e}")
    
    return examples

def train_model(model, tokenizer, train_dataset, eval_dataset, lora_config, output_dir, batch_size=4, num_epochs=3, gradient_accumulation_steps=8):
    """Fine-tune the model using LoRA."""
    logger.info("Setting up LoRA for training")
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
        target_modules=lora_config["target_modules"],
    )
    
    # Apply LoRA config to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=100,
        logging_steps=10, # Evaluation will happen at logging steps
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        group_by_length=True,
        report_to="tensorboard",
    )
    
    # Set up the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # Remove compute_metrics to avoid OOM error during evaluation
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Evaluate the model on the evaluation set
    logger.info("Starting final evaluation on the evaluation dataset")
    eval_results = trainer.evaluate()
    # Log the eval_loss which is computed batch by batch (memory-efficient)
    logger.info(f"Evaluation Loss: {eval_results.get('eval_loss', 'N/A')}")
    
    # Generate and log sample predictions for qualitative analysis
    examples = generate_example_predictions(model, tokenizer, eval_dataset, num_examples=5)
    
    logger.info("Training completed successfully!")

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description="Train a LoRA-adapted Phi-2 agent for medical diagnosis")
    
    # Agent configuration
    parser.add_argument("--agent", type=str, required=True, help="Name of the agent (e.g., cardio, metabolic)")
    parser.add_argument("--allow_repeats", action="store_true", help="Allow repeating diagnoses in dataset")
    
    # Model paths
    # --- IMPORTANT: Replace with your actual path or use environment variable ---
    # default_base_model_path = r"A:\201_HuggingFace\models--microsoft--phi-2\snapshots\ef382358ec9e382308935a992d908de099b64c23" # Example of hardcoded path (BAD)
    default_base_model_path = os.environ.get("DEFAULT_PHI2_BASE_PATH", "path/to/your/phi-2/model") # Recommended approach
    parser.add_argument("--base_model", type=str,
                        default=default_base_model_path,
                        help="Path to base model directory (or set DEFAULT_PHI2_BASE_PATH env var)")
    
    # Data path
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="Path to domain-specific data directory (default: data/arrow/<agent>)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout rate")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default data directory based on agent name if not specified
    if args.data_dir is None:
        agent_data_dir = DATA_DIR / args.agent
        if agent_data_dir.exists():
            args.data_dir = str(agent_data_dir)
        else:
            # Fall back to general data directory
            args.data_dir = str(DATA_DIR / "general")
            if not (DATA_DIR / "general").exists():
                args.data_dir = str(DATA_DIR)  # Last resort: use base arrow directory
    
    # Set up logging for this agent
    setup_logging(args.agent)
    
    # Load and update config
    config = load_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}
    config = update_config_with_args(config, args)
    agent_config = config['agents'][args.agent]
    
    # Print configuration
    logger.info(f"Configuration for agent {args.agent}:")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(json.dumps(agent_config, indent=2))
    
    # Prepare model and tokenizer using the path provided in args directly
    model, tokenizer = prepare_model_and_tokenizer(args.base_model)
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset(args.data_dir, agent_config['allow_repeats'])
    
    # Train model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lora_config=agent_config['lora'],
        output_dir=agent_config['output_dir'],
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
    )

if __name__ == "__main__":
    main() 