"""
Author: Tim Frenzel
Version: 1.00
Usage:  python src/download_model.py

Objective of the Code:
------------
Downloads the base Phi-2 language model (or other specified model) from
Hugging Face Hub, applies BitsAndBytes 4-bit quantization configuration,
and saves the quantized model and tokenizer to a local directory 
(typically defined by HF_HOME or a specific cache/save path).
It includes post-download checks to verify model loading and basic functionality.

"""

# src/download_model.py
import sys
import importlib
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Check for required packages
required_packages = {
    "torch": "pytorch",  # Conda/pip install name might differ
    "transformers": "transformers",
    "bitsandbytes": "bitsandbytes"
}
missing_packages = []

for package_name, install_name in required_packages.items():
    try:
        importlib.import_module(package_name)
        print(f"Package '{package_name}' found.")
    except ImportError:
        missing_packages.append(f"{package_name} (install with: pip install {install_name} or check conda install)")
    except Exception as e:
        if "numpy" in str(e).lower():
             print(f"Warning: Potential NumPy version conflict during {package_name} import: {e}")
             print("Attempting to continue...")
        else:
            raise e

if missing_packages:
    print("\nError: The following required packages are missing or could not be imported:")
    for pkg in missing_packages:
        print(f"- {pkg}")
    print("\nPlease ensure the 'phi2_moa' environment is active and all dependencies are installed correctly.")
    sys.exit(1) # Exit the script if packages are missing

# If all packages are present, proceed with the original script logic
print("\nAll required packages found. Proceeding with model download...")

# Define the target directory for both cache and final saves
# Use environment variable HF_HOME if set, otherwise default to ~/.cache/huggingface
cache_dir_base = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
model_name = "microsoft/phi-2"

# --- Define the SINGLE target save directory based on the standard cache structure ---
# This assumes the snapshot hash is known or can be determined, using the known one for now.
snapshot_hash = "ef382358ec9e382308935a992d908de099b64c23"
save_path = os.path.join(cache_dir_base, f"models--{model_name.replace('/', '--')}", "snapshots", snapshot_hash)

# Create the target directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print(f"Using directory for cache: {cache_dir_base}")
print(f"Model and Tokenizer will be saved to: {save_path}")

# --- Load Tokenizer ---
print("Loading tokenizer...")
try:
    tok = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir_base # Explicitly set cache dir
    )
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

# --- Configure Quantization ---
# Use the recommended BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Use torch dtype directly
    # bnb_4bit_use_double_quant=True, # Optional: Can improve precision slightly
    # bnb_4bit_quant_type="nf4" # Optional: Default is usually fine
)

# --- Load Model ---
print("Loading model with quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config, # Re-enabled after library update
        device_map={"": 0}, # Explicitly map to GPU 0
        cache_dir=cache_dir_base
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- Save ---
print("Saving tokenizer...")
tok.save_pretrained(save_path) # Save to the single target directory
print("Saving model...")
model.save_pretrained(save_path) # Save to the single target directory
print("Model and tokenizer saved successfully.")


# === Post-Loading Tests ===
print("\n=== Running Post-Loading Tests ===")

# 1. Inspect Model Properties (Check dtype and device)
print("\n--- 1. Inspecting Model Properties ---")
try:
    # Accessing weights might differ slightly based on model structure
    # Let's try accessing the embedding layer weights
    sample_weight = model.get_input_embeddings().weight
    print(f"Sample weight tensor dtype: {sample_weight.dtype}")
    print(f"Sample weight tensor device: {sample_weight.device}")
    # For quantized models, the compute dtype is also relevant
    if hasattr(model, 'quantization_config') and model.quantization_config.load_in_4bit:
        print(f"Quantization compute dtype: {model.quantization_config.bnb_4bit_compute_dtype}")
except Exception as e:
    print(f"Error inspecting model properties: {e}")

# 2. Tokenizer Check
print("\n--- 2. Tokenizer Check ---")
try:
    test_sentence = "Hello world! This is a test."
    print(f"Original sentence: {test_sentence}")
    encoded_input = tok(test_sentence, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    print(f"Tokenized Input IDs: {input_ids}")
    decoded_text = tok.decode(input_ids[0], skip_special_tokens=True)
    print(f"Decoded sentence: {decoded_text}")
    if test_sentence == decoded_text:
        print("Tokenizer encode/decode successful.")
    else:
        print("Warning: Tokenizer encode/decode mismatch.")
except Exception as e:
    print(f"Error during tokenizer check: {e}")

# 3. Simple Text Generation
print("\n--- 3. Simple Text Generation ---")
prompt = "The capital of France is"
print(f"Prompt: '{prompt}'")
try:
    inputs = tok(prompt, return_tensors="pt", return_attention_mask=False)
    # Move inputs to the same device as the model (important if device_map wasn't used or failed)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    outputs = model.generate(**inputs, max_length=20) # Generate a few tokens
    text_output = tok.batch_decode(outputs)[0]
    print(f"Generated text: {text_output}")
except Exception as e:
    print(f"Error during text generation: {e}")

# 4. Basic Inference Speed Test
print("\n--- 4. Basic Inference Speed Test ---")
import time
num_iterations = 3
total_time = 0
prompt_speed_test = "Once upon a time,"
try:
    inputs_speed = tok(prompt_speed_test, return_tensors="pt", return_attention_mask=False)
    inputs_speed = {key: value.to(model.device) for key, value in inputs_speed.items()}
    
    print(f"Running {num_iterations} generation iterations (max_length=15)...")
    for i in range(num_iterations):
        start_time = time.time()
        _ = model.generate(**inputs_speed, max_length=15) # Generate a fixed small number of tokens
        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time
        print(f"Iteration {i+1} time: {iteration_time:.4f} seconds")
    
    average_time = total_time / num_iterations
    print(f"Average generation time: {average_time:.4f} seconds")
except Exception as e:
    print(f"Error during inference speed test: {e}")

# 5. Configuration Verification
print("\n--- 5. Configuration Verification ---")
try:
    print(f"Model Config Class: {model.config.__class__.__name__}")
    print(f"Model Type: {model.config.model_type}")
    print(f"Vocabulary Size: {model.config.vocab_size}")
    print(f"Number of Hidden Layers: {model.config.num_hidden_layers}")
except Exception as e:
    print(f"Error verifying configuration: {e}")

print("\n=== Post-Loading Tests Completed ===")
