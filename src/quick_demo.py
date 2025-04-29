"""
Author: Tim Frenzel
Version: 1.10
Usage:  python src/quick_demo.py --prompt "Patient presents with..." [--base_model_path <path>] [--checkpoints_dir <path>]

Objective of the Code:
------------
Provides a simple command-line interface (CLI) to demonstrate the end-to-end
Mixture-of-Agents (MoA) pipeline on a single patient prompt. It orchestrates
the call sequence: prompt -> router -> specialist agents (loaded) ->
refinement (mocked/real) -> consensus -> final diagnosis output. Useful for
quick functional tests and qualitative evaluation.
"""

import os
import sys
import argparse
import pathlib
import logging
import warnings

# --- Add project root to sys.path --- 
# This allows importing modules from src/
ROOT = pathlib.Path(__file__).resolve().parents[1] # Go up one level from src/
sys.path.append(str(ROOT))

# --- Project Modules --- 
try:
    from src.router import Router
    from src.refine import Refiner
    from src.consensus import ConsensusAggregator
    # Model loading/generation dependencies
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
except ImportError as e:
    print(f"Error importing project modules or dependencies: {e}")
    print("Ensure src/router.py, src/refine.py, src/consensus.py exist and required libraries (torch, transformers, peft, bitsandbytes) are installed.")
    sys.exit(1)

# --- Configuration --- 
# Default paths - adjust if your structure differs
DEFAULT_ROUTER_CONFIG = ROOT / "domain_map.yaml"
DEFAULT_CHECKPOINTS_DIR = ROOT / "checkpoints" # Consistent with train_agent.py output
DEFAULT_REFINE_PROMPT = ROOT / "prompts" / "aggregate.txt"
# --- IMPORTANT: Base model path needs to be set correctly ---
# Recommend providing via --base_model_path argument or setting DEFAULT_PHI2_BASE_PATH env var.
DEFAULT_BASE_MODEL_PATH = os.environ.get("DEFAULT_PHI2_BASE_PATH", "path/to/your/phi-2/model") # Placeholder default

# --- Logging --- 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings if needed (e.g., from bitsandbytes)
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# --- Helper Function to Load Base Model and Tokenizer ---
@torch.inference_mode() # Ensure no gradients are computed
def load_base_model_and_tokenizer(model_path_or_id: str | pathlib.Path):
    """Loads the base Phi-2 model (quantized) and tokenizer."""
    logger.info(f"Loading base model and tokenizer from: {model_path_or_id}")
    model_path = pathlib.Path(model_path_or_id)
    if not model_path.exists():
        logger.error(f"Base model path does not exist: {model_path}")
        raise FileNotFoundError(f"Base model path not found: {model_path}")

    try:
        # Quantization Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16, # Consistent dtype
            device_map="auto", # Automatically distribute across available devices
            trust_remote_code=True,
            low_cpu_mem_usage=True # Optimization for loading large models
        )
        
        # Add padding token if missing (Phi-2 doesn't always have one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # No need to resize embeddings if pad_token is just set to eos_token
            # model.resize_token_embeddings(len(tokenizer))
            logger.info("Set PAD token to EOS token.")
            
        logger.info("Base model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load base model/tokenizer from {model_path}: {e}", exc_info=True)
        raise RuntimeError(f"Could not load base model from {model_path}") from e

@torch.inference_mode() # Ensure no gradients are computed
def predict_specialist(model: PeftModel, tokenizer: AutoTokenizer, prompt: str, device: str) -> tuple[str, float]:
    """Generates a diagnosis using the loaded specialist agent (LoRA adapted model)."""
    # Prepare input
    # Note: Check if fine-tuning used a specific prompt format (e.g., Instruct) - adapt here if necessary.
    # Assuming simple prompt for now.
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024).to(device)
    
    # Generation parameters (adjust as needed)
    max_new_tokens = 50 # Limit generated output length
    temperature = 0.1 # Lower temperature for less random outputs
    top_p = 0.9
    do_sample=True

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id # Use the defined pad token
        )
        
        # Decode output, skipping the prompt tokens
        # Important: outputs[0] contains the full sequence (prompt + generation)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        diagnosis = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Placeholder confidence - real confidence estimation is complex
        confidence = 0.8 
        logger.warning("Assigned placeholder confidence score (0.8) to specialist prediction.")
        
        return diagnosis, confidence

    except Exception as e:
        logger.error(f"Error during model generation: {e}", exc_info=True)
        return "Error during generation", 0.0

def run_pipeline(prompt: str, router_config_path: str, checkpoints_dir: str, base_model_path: str, refine_prompt_path: str, openai_api_key: str | None):
    """Executes the MoA pipeline for a single prompt."""
    logger.info("--- Starting MoA Pipeline Demo ---")
    logger.info(f"Input Prompt: {prompt[:100]}..." )

    # 0. Load Base Model and Tokenizer (once)
    try:
        base_model, tokenizer = load_base_model_and_tokenizer(base_model_path)
        device = base_model.device # Get device model was loaded onto
        logger.info(f"Base model loaded onto device: {device}")
    except Exception as e:
        logger.error(f"Fatal error loading base model: {e}")
        return None

    # 1. Router
    logger.info("--- Layer 1: Routing --- " )
    router_config_file = pathlib.Path(router_config_path)
    if not router_config_file.exists():
        logger.error(f"Router config not found: {router_config_file}")
        return None
    
    try:
        router = Router(config_path=router_config_file)
        selected_agent_ids = router.predict(prompt)
        logger.info(f"Router selected agents: {selected_agent_ids}")
    except Exception as e:
        logger.error(f"Error initializing or running Router: {e}", exc_info=True)
        return None

    # 2. Specialist Agents (Load and Predict)
    logger.info("--- Layer 1: Specialist Inference ---" )
    specialist_results = []
    checkpoints_path = pathlib.Path(checkpoints_dir)
    if not checkpoints_path.exists():
        logger.error(f"Checkpoints directory {checkpoints_path} not found. Cannot load specialist agents.")
        # Decide behavior: proceed without specialists or exit?
        # Proceeding without for demo, refinement layer will be the only input to consensus.
        selected_agent_ids = [] # Clear selected agents if dir missing
    elif not selected_agent_ids:
        logger.warning("Router did not select any specialist agents. Proceeding without specialist input.")
    else:
        # Load and predict with each selected agent
        for agent_id in selected_agent_ids:
            adapter_path = checkpoints_path / agent_id
            logger.info(f"Attempting to load adapter for agent '{agent_id}' from {adapter_path}")

            if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
                logger.warning(f"Adapter path or config not found for {agent_id} at {adapter_path}. Skipping agent.")
                continue

            try:
                # Load the LoRA adapter onto the base model
                specialist_model = PeftModel.from_pretrained(base_model, str(adapter_path))
                specialist_model.eval() # Set to evaluation mode
                logger.info(f"Successfully loaded adapter for '{agent_id}'")
                
                # Generate prediction
                diagnosis, confidence = predict_specialist(specialist_model, tokenizer, prompt, device)
                logger.info(f"Agent '{agent_id}' suggests: {diagnosis} (Conf: {confidence:.2f})")
                
                specialist_results.append({
                    "agent_id": agent_id,
                    "diagnosis": diagnosis,
                    "confidence": confidence
                })
                
                # Clear memory (important when loading multiple adapters sequentially)
                del specialist_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to load or run agent '{agent_id}' from {adapter_path}: {e}", exc_info=True)
                # Optionally add an error result
                # specialist_results.append({"agent_id": agent_id, "diagnosis": "Error", "confidence": 0.0})

    # 3. Refinement Layer
    logger.info("--- Layer 2: Refinement --- " )
    refine_prompt_file = pathlib.Path(refine_prompt_path)
    if not refine_prompt_file.exists():
        logger.error(f"Refinement prompt not found: {refine_prompt_file}")
        # Pipeline continues without refinement, consensus layer will handle it.
        refined_result = None 
    else:
        try:
            refiner = Refiner(api_key=openai_api_key, system_prompt_path=refine_prompt_file)
            use_mock = openai_api_key is None
            if use_mock:
                logger.warning("OpenAI API key not provided. Using MOCK refinement.")

            refined_diagnosis, refined_rationale = refiner.refine_diagnosis(
                original_prompt=prompt,
                agent_outputs=specialist_results,
                mock=use_mock # Use mock=True if key is missing
            )
            
            # Placeholder confidence - refine.py might need updating to provide this
            refined_confidence = 0.90 
            logger.warning("Assigned placeholder confidence score (0.90) to refinement result.")
            logger.info(f"Refined Diagnosis: {refined_diagnosis}")
            logger.info(f"Refinement Rationale: {refined_rationale}")
            refined_result = {
                "agent_id": "refiner_gpt3.5_turbo",
                "diagnosis": refined_diagnosis,
                "confidence": refined_confidence
            }
        except Exception as e:
            logger.error(f"Error during refinement layer: {e}", exc_info=True)
            refined_result = None # Ensure it's None on error

    # 4. Consensus Layer
    logger.info("--- Layer 3: Consensus --- " )
    try:
        aggregator = ConsensusAggregator()
        
        all_results = specialist_results
        if refined_result: # Only add refinement if it ran successfully
            all_results.append(refined_result)

        if not all_results:
            logger.error("No results from Layer 1 or Layer 2 to aggregate.")
            final_diagnosis = "Error: No agent outputs available for consensus."
        else:
            # Run both methods for demo purposes
            final_diagnosis_softmax, final_rationale_softmax = aggregator.aggregate(
                predictions=all_results, 
                method='softmax_weighted' # Use actual enum or string
            )
            logger.info(f"Final Diagnosis (Softmax): {final_diagnosis_softmax}")
            logger.info(f"Final Rationale (Softmax): {final_rationale_softmax}")

            final_diagnosis_majority, final_rationale_majority = aggregator.aggregate(
                predictions=all_results,
                method='majority' # Use actual enum or string
            )
            logger.info(f"Final Diagnosis (Majority): {final_diagnosis_majority}")
            
            final_diagnosis = final_diagnosis_softmax # Return softmax by default
    except Exception as e:
        logger.error(f"Error during consensus layer: {e}", exc_info=True)
        final_diagnosis = "Error during consensus aggregation."

    logger.info("--- MoA Pipeline Demo Complete ---")
    return final_diagnosis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MoA pipeline for a single prompt using loaded specialist agents.")
    parser.add_argument("--prompt", type=str, required=True, help="The patient prompt text.")
    parser.add_argument("--router_config", type=str, default=str(DEFAULT_ROUTER_CONFIG),
                        help="Path to the router's domain map YAML file.")
    parser.add_argument("--checkpoints_dir", type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
                        help="Directory containing specialist LoRA adapter checkpoints.")
    parser.add_argument("--base_model_path", type=str, default=str(DEFAULT_BASE_MODEL_PATH),
                        help="Path to the base Phi-2 model directory (must contain model and tokenizer files).")
    parser.add_argument("--refine_prompt", type=str, default=str(DEFAULT_REFINE_PROMPT),
                        help="Path to the refinement layer's system prompt.")
    parser.add_argument("--openai_key", type=str, default=os.environ.get("OPENAI_API_KEY"), 
                        help="OpenAI API key (or set OPENAI_API_KEY env var). If absent, uses mock refinement.")
    
    args = parser.parse_args()

    # Basic validation for base model path existence before running
    if not pathlib.Path(args.base_model_path).exists():
        logger.error(f"Base model path specified does not exist: {args.base_model_path}")
        logger.error("Please provide a valid path using --base_model_path or ensure the default path is correct.")
        sys.exit(1)

    run_pipeline(
        prompt=args.prompt,
        router_config_path=args.router_config,
        checkpoints_dir=args.checkpoints_dir,
        base_model_path=args.base_model_path, # Pass base model path
        refine_prompt_path=args.refine_prompt,
        openai_api_key=args.openai_key
    ) 