#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.50
Usage:  
    python src/evaluate.py --eval_path data/eval.arrow \
                           --router_config domain_map.yaml \
                           --checkpoints_dir checkpoints/ \
                           --openai_key <YOUR_KEY> \
                           --output results/moa_metrics.json

Objective of the Code:
------------
Runs the full end-to-end Mixture-of-Agents (MoA) pipeline evaluation. 
It loads an evaluation dataset (Arrow format), processes each prompt through 
the router, specialist agents (loaded from checkpoints), refinement layer 
(GPT-3.5), and consensus layer. Finally, it computes and saves diagnostic 
performance metrics (e.g., Accuracy, F1) comparing pipeline predictions 
against gold standard diagnoses.
"""
import os
import sys
import json
import argparse
import pathlib
import logging
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset

# Local imports
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from router import Router

# --- Print File Path --- >
print(f"[DEBUG] Executing script: {os.path.abspath(__file__)}")
# < --- End Print File Path ---

# --- Paths & Defaults (Defined Before Use) ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DEFAULT_AGENTS_DIR = MODELS_DIR
DEFAULT_BASE_MODEL_NAME = "microsoft/phi-2" # Default model ID
# --- IMPORTANT: Update base path based on your environment ---
# Option 1: Set Environment Variable (Recommended)
#   Set DEFAULT_PHI2_BASE_PATH in your system environment
# Option 2: Hardcode (Update this line if needed, but use env var preferably)
#   BASE_MODEL_DIR = pathlib.Path(r"A:\your\path\to\phi-2-snapshot") 
BASE_MODEL_DIR = pathlib.Path(os.environ.get("DEFAULT_PHI2_BASE_PATH", str(ROOT / "models" / "base-phi-2-placeholder")))

# --- Configure logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluator class for measuring agent performance."""
    
    def __init__(self, base_model_path=None, target_agent=None, device=None, agents_dir=None):
        """Initialize the evaluator, setting up either a single agent or the router."""
        logger.info("Initializing medical diagnosis evaluator")

        # --- Determine Mode and Paths ---
        self.target_agent = target_agent
        self.single_agent_mode = bool(self.target_agent)

        # Base Model Path: Use provided path, or default, convert to Path
        raw_base_path = base_model_path if base_model_path else BASE_MODEL_DIR
        try:
            # Attempt conversion to Path, handle potential errors if path is invalid type initially
            self.base_model_path = pathlib.Path(raw_base_path)
            logger.info(f"Using base model path: {self.base_model_path}")
            if not self.base_model_path.exists():
                logger.error(f"Resolved base model path does not exist: {self.base_model_path}")
                raise FileNotFoundError(f"Base model path not found: {self.base_model_path}")
        except TypeError as e:
             logger.error(f"Invalid type provided for base_model_path: {raw_base_path}. Error: {e}")
             raise ValueError(f"Invalid base_model_path: {raw_base_path}") from e


        # Agents Directory: Use provided path, or default, convert to Path
        raw_agents_dir = agents_dir if agents_dir else DEFAULT_AGENTS_DIR # Assuming DEFAULT_AGENTS_DIR is defined
        try:
             self.agents_dir = pathlib.Path(raw_agents_dir)
             logger.info(f"Using agents directory: {self.agents_dir}")
             # Optional: Check if agents_dir exists if needed immediately
             # if not self.agents_dir.exists():
             #    logger.warning(f"Agents directory does not exist: {self.agents_dir}")
        except TypeError as e:
             logger.error(f"Invalid type provided for agents_dir: {raw_agents_dir}. Error: {e}")
             raise ValueError(f"Invalid agents_dir: {raw_agents_dir}") from e


        # Device: Use provided device or detect automatically
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize router or single agent attributes to None initially
        self.router = None
        self.agent_model = None
        self.agent_tokenizer = None

        # --- Conditional Initialization ---
        if self.single_agent_mode:
            logger.info(f"Initializing in Single Agent Mode for agent: '{self.target_agent}'")
            self._load_single_agent() # Load the specific agent model/tokenizer
        else:
            logger.info("Initializing in Router Mode (multi-agent evaluation)")
            if Router:
                logger.info("Router class available, initializing Router.")
                # Pass paths as strings to Router if it expects strings
                self.router = Router(
                    base_model_path=str(self.base_model_path),
                    agents_dir=str(self.agents_dir),
                    device=self.device
                )
                # Log router info
                if hasattr(self.router, 'agents') and self.router.agents:
                     logger.info(f"Router initialized with {len(self.router.agents)} agents: {', '.join(self.router.agents.keys())}")
                     # Add checks for 'general' agent presence as before
                     if "general" not in self.router.agents and len(self.router.agents) == 0:
                          logger.warning("Router initialized, but no base model ('general') or specialized agents were loaded successfully.")
                     elif "general" not in self.router.agents:
                          logger.warning("Router initialized, specialized agents found, but base model ('general') failed to load.")
                     elif len(self.router.agents) == 1 and "general" in self.router.agents:
                          logger.warning("Router initialized, but only the base model ('general') was loaded. No specialized agents found or loaded.")
                else:
                     logger.warning("Router object initialized, but no agents seem to be loaded.")

            else:
                logger.error("Router class not found or failed to import. Router mode disabled.")
                raise ImportError("Router class is required for multi-agent evaluation but could not be imported.")

        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Evaluator initialization complete.")

    def _load_single_agent(self):
        """Loads the model and tokenizer for the specified single agent."""
        if not self.target_agent:
            raise ValueError("Cannot load single agent without target_agent being set.")

        # Corrected: Append '_lora' to the target agent name for path construction
        agent_dir_name = f"{self.target_agent}_lora"
        agent_path = self.agents_dir / agent_dir_name
        logger.info(f"Attempting to load agent model from: {agent_path}")

        if not agent_path.exists():
             logger.error(f"Agent directory not found: {agent_path}")
             # Use agent_dir_name in the error message for clarity
             raise FileNotFoundError(f"Could not find agent directory '{agent_dir_name}' at {agent_path}")

        try:
            # Load base model tokenizer (usually shared)
            self.agent_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            self.agent_tokenizer.pad_token = self.agent_tokenizer.eos_token # Common practice

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                return_dict=True,
                torch_dtype=torch.float16, # Or adjust dtype as needed
                trust_remote_code=True,
                # device_map='auto' # Consider device_map for large models
            )
            base_model.to(self.device) # Move base model to device first

            # Load PEFT model (adapter) on top of the base model
            logger.info(f"Loading PEFT adapter for '{self.target_agent}' from {agent_path}")
            self.agent_model = PeftModel.from_pretrained(base_model, str(agent_path))
            # No need to move PeftModel again if base model is on the correct device

            self.agent_model.eval() # Set model to evaluation mode
            logger.info(f"Successfully loaded agent '{self.target_agent}' onto device '{self.device}'.")

        except ImportError as e:
             logger.error(f"ImportError during model loading for {self.target_agent}. Do you have 'peft' installed? Error: {e}")
             raise
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer for agent '{self.target_agent}' from {agent_path}: {e}", exc_info=True)
            raise RuntimeError(f"Could not load agent '{self.target_agent}'.") from e

    def evaluate_dataset(self, dataset_path, target_agent=None, snomed_filter=None, limit=None):
        """Evaluate agent(s) using a test dataset."""
        logger.info(f"Starting evaluation on dataset: {dataset_path}")
        if target_agent:
            logger.info(f"Evaluating specific agent: {target_agent}")
        if snomed_filter:
            logger.info(f"Filtering dataset with SNOMED CT prefix: {snomed_filter}")

        try:
            # Load the dataset (assuming Hugging Face datasets format)
            test_dataset = load_dataset("json", data_files=dataset_path, split="train")
            logger.info(f"Loaded dataset with {len(test_dataset)} samples.")
        except Exception as e:
            logger.error(f"Failed to load dataset from {dataset_path}: {e}")
            return {}

        true_diagnoses = []
        pred_diagnoses = []
        confidences = []
        selected_agents = []
        processing_times = []
        skipped_count = 0

        # Filter dataset if snomed_filter is provided
        if snomed_filter:
            original_size = len(test_dataset)
            def filter_func(example):
                # Assuming the ground truth code is in 'diagnosis_code' or similar field
                # Adjust the key based on your actual dataset structure
                true_code = example.get('diagnosis_code') or example.get('label')
                if true_code and isinstance(true_code, str):
                    return true_code.startswith(snomed_filter)
                return False
            test_dataset = test_dataset.filter(filter_func)
            logger.info(f"Filtered dataset from {original_size} to {len(test_dataset)} samples using SNOMED CT prefix '{snomed_filter}'.")
            if len(test_dataset) == 0:
                logger.warning("No samples remaining after filtering. Evaluation cannot proceed.")
                return {}

        for i, sample in enumerate(test_dataset):
            start_time = time.time()
            try:
                # Extract patient data and true diagnosis (SNOMED CT) from sample
                # Adjust keys based on your dataset structure
                patient_data = sample.get("patient_data") # Assuming data is structured this way
                true_diagnosis_code = sample.get('diagnosis_code') or sample.get('label')

                if not patient_data or not true_diagnosis_code:
                     # Attempt to parse from 'text' if structured data is missing
                    text = sample.get("text", "")
                    if text:
                        parsed_data = self._parse_text_sample(text)
                        patient_data = parsed_data.get("patient_data", {})
                        true_diagnosis_code = parsed_data.get("true_diagnosis")

                if not patient_data:
                    logger.warning(f"Sample {i} missing patient data, skipping")
                    skipped_count += 1
                    continue

                if not true_diagnosis_code:
                    logger.warning(f"Sample {i} missing true diagnosis code (SNOMED CT), skipping")
                    skipped_count += 1
                    continue

                # Add the true diagnosis code to the patient data if needed by the router/agents
                # patient_data['true_diagnosis_code'] = true_diagnosis_code

                # --- Agent Routing and Prediction --- #
                if self.single_agent_mode:
                    # --- Single Agent Prediction ---
                    # Build prompt (you might need a dedicated prompt builder or reuse router's logic)
                    # Example placeholder - adapt based on actual prompt needs
                    prompt = f"Patient Data: {patient_data}\nDiagnose:" # Corrected: Closing quote moved

                    if not prompt:
                         logger.warning(f"Sample {i}: Failed to build prompt, skipping.")
                         skipped_count += 1
                         continue

                    try:
                        # Generate diagnosis using self.agent_model and self.agent_tokenizer
                        inputs = self.agent_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device) # Adjust max_length
                        with torch.no_grad():
                             # Note: Generation parameters might need adjustment
                             outputs = self.agent_model.generate(**inputs, max_new_tokens=50, pad_token_id=self.agent_tokenizer.eos_token_id)
                        # Decode the output (excluding prompt)
                        raw_output = self.agent_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        # --- Post-process raw_output to extract SNOMED code and confidence ---
                        # This part is highly dependent on your model's output format
                        # Placeholder: Assume output is just the code for now
                        predicted_code = raw_output.strip() # Needs robust parsing
                        confidence = 1.0 # Placeholder - confidence calculation needed
                        used_agent = self.target_agent
                        # ----------------------------------------------------------------------
                    except Exception as gen_e:
                         logger.error(f"Sample {i}: Error during generation for single agent '{self.target_agent}': {gen_e}", exc_info=True)
                         # Set results to indicate failure for this sample
                         predicted_code = "ERROR_GENERATION"
                         confidence = 0.0
                         used_agent = self.target_agent
                         # No need to skip count here, we store the error indication

                else: # Router mode (existing logic)
                    # --- Router Prediction ---
                    router_result = self.router.route(patient_data)
                    predicted_code = router_result.get("snomed_code") # Expecting snomed_code now
                    confidence = router_result.get("confidence", 0.0)
                    used_agent = router_result.get("agent")
                # ------------------------------------ #

                if not predicted_code:
                     logger.warning(f"Sample {i} did not yield a predicted code, skipping storage.")
                     # Or store as None/empty if you want to count prediction failures
                     # predicted_diagnoses.append(None)
                     # confidences.append(0.0)
                     # selected_agents.append(used_agent)
                     # true_diagnoses.append(true_diagnosis_code)
                     skipped_count += 1
                     continue

                # Store results
                true_diagnoses.append(str(true_diagnosis_code)) # Ensure string comparison
                pred_diagnoses.append(str(predicted_code))     # Ensure string comparison
                confidences.append(confidence)
                selected_agents.append(used_agent)
                processing_times.append(time.time() - start_time)

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}", exc_info=True)
                skipped_count += 1

            if limit and i >= limit - skipped_count: # Adjust limit check based on processed samples
                 logger.info(f"Reached limit of {limit} processed samples.")
                 break

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} samples due to missing data or errors.")

        # Check if any predictions were actually made
        if not pred_diagnoses:
            logger.warning("No predictions were generated. Cannot calculate metrics.")
            return {
                "metadata": {
                    "dataset": str(dataset_path),
                    "evaluated_agent": self.target_agent if self.single_agent_mode else "Router",
                    "snomed_filter": snomed_filter,
                    "limit": limit,
                    "total_samples_in_dataset_after_filter": len(test_dataset) if 'test_dataset' in locals() else 'N/A', # Handle case where dataset loading failed
                    "processed_samples": 0,
                    "skipped_samples": skipped_count,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "evaluation_time_seconds": sum(processing_times),
                    "avg_processing_time_per_sample": 0
                },
                "predictions": [],
                "metrics": {}
            }

        # --- Calculate Metrics --- #
        logger.info(f"Calculating metrics for {len(pred_diagnoses)} predictions.")
        try:
            accuracy = accuracy_score(true_diagnoses, pred_diagnoses)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_diagnoses, pred_diagnoses, average='weighted', zero_division=0
            )
            # Example: Add macro average F1 as well
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_diagnoses, pred_diagnoses, average='macro', zero_division=0
            )
        except Exception as metrics_e:
            logger.error(f"Error calculating metrics: {metrics_e}", exc_info=True)
            # Handle case where metrics calculation fails
            accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
            f1_macro = 0.0 # Initialize macro F1

        # Agent usage counts (more relevant in Router mode)
        agent_counts = defaultdict(int)
        for agent in selected_agents:
             agent_counts[agent] += 1

        # --- Prepare Results Dictionary --- #
        results = {
            "metadata": {
                "dataset": str(dataset_path),
                "evaluated_agent": self.target_agent if self.single_agent_mode else "Router",
                "snomed_filter": snomed_filter,
                "limit": limit,
                "total_samples_in_dataset_after_filter": len(test_dataset) if 'test_dataset' in locals() else 'N/A',
                "processed_samples": len(pred_diagnoses),
                "skipped_samples": skipped_count,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_time_seconds": sum(processing_times),
                "avg_processing_time_per_sample": np.mean(processing_times) if processing_times else 0
            },
            "metrics": {
                "overall": {
                    "accuracy": accuracy,
                    "precision_weighted": precision,
                    "recall_weighted": recall,
                    "f1_score_weighted": f1,
                    "f1_score_macro": f1_macro # Add macro F1
                },
                "agent_usage": dict(agent_counts) # Store agent counts
            },
            "predictions": [
                {
                    # Using idx from enumerate for sample index consistency
                    "sample_index": idx,
                    "true_diagnosis": true,
                    "predicted_diagnosis": pred,
                    "confidence": conf,
                    "agent_used": agent,
                    "correct": str(true) == str(pred)
                }
                for idx, (true, pred, conf, agent) in enumerate(zip(true_diagnoses, pred_diagnoses, confidences, selected_agents))
            ]
        }

        # Log key metrics
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Overall F1 Score (Weighted): {f1:.4f}")
        if not self.single_agent_mode:
             logger.info(f"Agent Usage Counts: {dict(agent_counts)}")

        # --- Save and Visualize Results --- #
        self._save_results(results)
        self._generate_visualizations_from_results(results) # Adapt visualization call

        return results

    def evaluate_jsonl(self, jsonl_path, target_agent=None):
        """Evaluate agent(s) using a JSONL file of test cases."""
        logger.info(f"Evaluating with JSONL file: {jsonl_path}")
        
        # Load test cases
        test_cases = []
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    test_cases.append(json.loads(line))
            
            logger.info(f"Loaded {len(test_cases)} test cases")
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            return None
        
        # Prepare results storage
        results = {
            "metadata": {
                "dataset": str(jsonl_path),
                "target_agent": target_agent,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "samples": len(test_cases)
            },
            "predictions": [],
            "metrics": {}
        }
        
        # For each test case
        true_diagnoses = []
        pred_diagnoses = []
        confidences = []
        selected_agents = []
        
        start_time = time.time()
        
        for i, case in enumerate(test_cases):
            try:
                # Extract patient data
                patient_data = case.get("patient_data", {})
                true_diagnosis = case.get("true_diagnosis")
                
                if not true_diagnosis:
                    logger.warning(f"Case {i} missing true diagnosis, skipping")
                    continue
                
                # Use specific agent if specified, otherwise use router
                if target_agent and target_agent in self.router.agents:
                    # Single agent evaluation
                    agent_model = self.router.agents[target_agent]["model"]
                    
                    # Build prompt for this agent
                    prompt = self.router._build_prompt(patient_data)
                    if not prompt:
                        logger.warning(f"Failed to build prompt for case {i}, skipping")
                        continue
                    
                    # Get confidence
                    confidence_data = self.router._agent_confidence(agent_model, prompt)
                    
                    # Generate diagnosis
                    diagnosis = self.router._generate_diagnosis(agent_model, prompt)
                    
                    predicted = diagnosis
                    confidence = confidence_data.get("confidence", 0)
                    used_agent = target_agent
                else:
                    # Use router for agent selection
                    router_result = self.router.route(patient_data)
                    
                    predicted = router_result.get("final_diagnosis")
                    confidence = router_result.get("agent_results", {}).get(
                        router_result.get("selected_agent", ""), {}).get("confidence", 0)
                    used_agent = router_result.get("selected_agent")
                
                # Store results
                true_diagnoses.append(true_diagnosis)
                pred_diagnoses.append(predicted)
                confidences.append(confidence)
                selected_agents.append(used_agent)
                
                # Add to detailed predictions
                results["predictions"].append({
                    "case_idx": i,
                    "case_id": case.get("id", str(i)),
                    "true_diagnosis": true_diagnosis,
                    "predicted_diagnosis": predicted,
                    "agent": used_agent,
                    "confidence": float(confidence) if isinstance(confidence, (int, float, np.number)) else 0,
                    "correct": str(predicted) == str(true_diagnosis)
                })
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(test_cases)} cases")
                    
            except Exception as e:
                logger.error(f"Error processing case {i}: {e}")
                continue
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        if pred_diagnoses and true_diagnoses:
            # Convert to strings for comparison
            pred_str = [str(p) for p in pred_diagnoses]
            true_str = [str(t) for t in true_diagnoses]
            
            # Calculate accuracy
            accuracy = accuracy_score(true_str, pred_str)
            
            # Calculate other metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_str, pred_str, average='weighted', zero_division=0
            )
            
            # Group by agent
            agent_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
            for i, agent in enumerate(selected_agents):
                agent_metrics[agent]["total"] += 1
                if str(pred_diagnoses[i]) == str(true_diagnoses[i]):
                    agent_metrics[agent]["correct"] += 1
            
            # Calculate per-agent accuracy
            for agent in agent_metrics:
                agent_metrics[agent]["accuracy"] = (
                    agent_metrics[agent]["correct"] / agent_metrics[agent]["total"]
                )
            
            # Save metrics
            results["metrics"] = {
                "overall": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "total_samples": len(true_diagnoses),
                    "evaluation_time_seconds": elapsed_time
                },
                "per_agent": {
                    agent: metrics for agent, metrics in agent_metrics.items()
                }
            }
            
            # Log metrics
            logger.info(f"Overall accuracy: {accuracy:.4f}")
            logger.info(f"Overall F1 score: {f1:.4f}")
            for agent, metrics in agent_metrics.items():
                logger.info(f"{agent} accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        
        # Generate visualizations
        self._generate_visualizations(results, jsonl_path)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_info = f"_{target_agent}" if target_agent else ""
        
        results_file = RESULTS_DIR / f"eval{target_info}_{timestamp}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        return results
    
    def _generate_visualizations(self, results, data_path):
        """Generate visualizations of evaluation results."""
        try:
            # Extract data for plotting
            predictions = results["predictions"]
            if not predictions:
                logger.warning("No predictions to visualize")
                return
            
            # Create a DataFrame for easier analysis
            df = pd.DataFrame(predictions)
            
            # Create results directory if it doesn't exist
            viz_dir = RESULTS_DIR / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            data_name = pathlib.Path(str(data_path)).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"{data_name}_{timestamp}"
            
            # VISUALIZATION 1: Accuracy by agent
            if "agent" in df.columns:
                plt.figure(figsize=(10, 6))
                agent_accuracy = df.groupby("agent")["correct"].mean()
                agent_accuracy.plot(kind="bar")
                plt.title("Accuracy by Agent")
                plt.ylabel("Accuracy")
                plt.xlabel("Agent")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / f"{prefix}_agent_accuracy.png")
                plt.close()
            
            # VISUALIZATION 2: Confidence vs. Correctness
            if "confidence" in df.columns and "correct" in df.columns:
                plt.figure(figsize=(10, 6))
                correct_conf = df[df["correct"]]["confidence"]
                incorrect_conf = df[~df["correct"]]["confidence"]
                
                plt.hist([correct_conf, incorrect_conf], bins=10, 
                         alpha=0.7, label=["Correct", "Incorrect"])
                plt.title("Confidence Distribution by Correctness")
                plt.xlabel("Confidence")
                plt.ylabel("Number of Predictions")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / f"{prefix}_confidence_dist.png")
                plt.close()
                
                # Calculate and visualize correlation between confidence and correctness
                plt.figure(figsize=(8, 6))
                conf_bins = np.linspace(0, 1, 11)
                bin_indices = np.digitize(df["confidence"], conf_bins) - 1
                bin_accuracy = []
                bin_counts = []
                
                for i in range(len(conf_bins) - 1):
                    mask = bin_indices == i
                    if mask.sum() > 0:
                        bin_accuracy.append(df.loc[mask, "correct"].mean())
                        bin_counts.append(mask.sum())
                    else:
                        bin_accuracy.append(0)
                        bin_counts.append(0)
                
                # Confidence calibration plot
                plt.scatter(
                    [(conf_bins[i] + conf_bins[i+1])/2 for i in range(len(conf_bins) - 1)],
                    bin_accuracy,
                    s=[min(c*20, 300) for c in bin_counts],
                    alpha=0.7
                )
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Perfect calibration line
                plt.xlabel("Confidence")
                plt.ylabel("Accuracy")
                plt.title("Confidence Calibration Plot")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / f"{prefix}_calibration.png")
                plt.close()
            
            logger.info(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    def _save_results(self, results):
        """Saves the evaluation results dictionary to a JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = pathlib.Path(results["metadata"]["dataset"]).stem
            agent_info = results["metadata"]["evaluated_agent"]
            results_file = RESULTS_DIR / f"eval_{dataset_name}_{agent_info}_{timestamp}.json"

            # Convert numpy types to standard Python types for JSON serialization
            def convert_numpy(obj):
                 if isinstance(obj, np.integer):
                      return int(obj)
                 elif isinstance(obj, np.floating):
                      return float(obj)
                 elif isinstance(obj, np.ndarray):
                      return obj.tolist()
                 elif isinstance(obj, pathlib.Path):
                      return str(obj)
                 return obj

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)

    def _generate_visualizations_from_results(self, results):
        """Generate visualizations based on the results dictionary."""
        try:
            # Extract data for plotting from the results dictionary
            predictions = results.get("predictions", [])
            metadata = results.get("metadata", {})
            metrics = results.get("metrics", {})

            if not predictions:
                logger.warning("No predictions found in results, cannot generate visualizations.")
                return

            # Create a DataFrame for easier analysis
            df = pd.DataFrame(predictions)

            # Create results directory if it doesn't exist
            viz_dir = RESULTS_DIR / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename prefix based on metadata
            dataset_name = pathlib.Path(metadata.get("dataset", "unknown_dataset")).stem
            agent_info = metadata.get("evaluated_agent", "unknown_agent")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"{dataset_name}_{agent_info}_{timestamp}"

            # --- Adapt Visualizations ---

            # VISUALIZATION 1: Accuracy by agent (Only makes sense in Router mode)
            if not self.single_agent_mode and "agent_used" in df.columns:
                agent_accuracy = df.groupby("agent_used")["correct"].mean()
                if not agent_accuracy.empty:
                    plt.figure(figsize=(10, 6))
                    agent_accuracy.plot(kind="bar")
                    plt.title("Accuracy by Agent Used (Router Mode)")
                    plt.ylabel("Accuracy")
                    plt.xlabel("Agent")
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"{prefix}_agent_accuracy.png")
                    plt.close()
                else:
                    logger.info("No agent usage data to plot accuracy by agent.")

            # VISUALIZATION 2: Confidence vs. Correctness (if confidence available)
            if "confidence" in df.columns and "correct" in df.columns and not df["confidence"].isnull().all():
                 # Ensure confidence is numeric, replace potential NaNs or non-numeric with 0
                 df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)

                 plt.figure(figsize=(10, 6))
                 correct_conf = df[df["correct"]]["confidence"]
                 incorrect_conf = df[~df["correct"]]["confidence"]

                 # Avoid plotting if one category is empty
                 hist_data = []
                 hist_labels = []
                 if not correct_conf.empty:
                      hist_data.append(correct_conf)
                      hist_labels.append("Correct")
                 if not incorrect_conf.empty:
                      hist_data.append(incorrect_conf)
                      hist_labels.append("Incorrect")

                 if hist_data:
                      plt.hist(hist_data, bins=10, alpha=0.7, label=hist_labels)
                      plt.title("Confidence Distribution by Correctness")
                      plt.xlabel("Confidence")
                      plt.ylabel("Number of Predictions")
                      plt.legend()
                      plt.grid(alpha=0.3)
                      plt.tight_layout()
                      plt.savefig(viz_dir / f"{prefix}_confidence_dist.png")
                      plt.close()
                 else:
                      logger.info("Not enough data to plot confidence distribution.")

                 # Confidence Calibration Plot (if enough data)
                 if len(df) > 10 and not df["confidence"].isnull().all(): # Need some data points
                    plt.figure(figsize=(8, 6))
                    # Define bins more robustly
                    min_conf, max_conf = df["confidence"].min(), df["confidence"].max()
                    # Ensure bins cover the range, handle case where min=max
                    if max_conf == min_conf:
                         conf_bins = np.linspace(min_conf - 0.05, min_conf + 0.05, 11)
                    else:
                         conf_bins = np.linspace(min_conf, max_conf, 11)

                    # Handle potential edge cases with digitize
                    # Ensure bin indices are within valid range [0, num_bins-1]
                    num_bins = len(conf_bins) - 1
                    bin_indices = np.digitize(df["confidence"], conf_bins, right=False) -1
                    bin_indices = np.clip(bin_indices, 0, num_bins -1)

                    bin_accuracy = []
                    bin_counts = []
                    bin_centers = []

                    for i in range(num_bins):
                         mask = bin_indices == i
                         count = mask.sum()
                         if count > 0:
                              bin_accuracy.append(df.loc[mask, "correct"].mean())
                              bin_counts.append(count)
                              bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)

                    if bin_centers: # Only plot if we have valid bins
                         # Scale point size by count, with min/max size limits
                         point_sizes = [min(max(c * 5, 20), 300) for c in bin_counts]
                         plt.scatter(
                              bin_centers,
                              bin_accuracy,
                              s=point_sizes,
                              alpha=0.7,
                              label="Bin Accuracy (Size ~ Count)"
                         )
                         plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect Calibration")
                         plt.xlabel("Confidence")
                         plt.ylabel("Accuracy")
                         plt.title("Confidence Calibration Plot")
                         plt.legend()
                         plt.grid(alpha=0.3)
                         # Adjust limits slightly beyond 0-1 if needed based on data
                         plt.xlim(-0.05, 1.05)
                         plt.ylim(-0.05, 1.05)
                         plt.tight_layout()
                         plt.savefig(viz_dir / f"{prefix}_calibration.png")
                         plt.close()
                    else:
                         logger.info("Not enough data points across bins to create calibration plot.")
                 else:
                      logger.info("Not enough data or confidence values missing/invalid for calibration plot.")
            else:
                 logger.info("Confidence or correctness data missing, skipping confidence visualizations.")

            logger.info(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)

# --- Command Line Interface ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate medical diagnosis agents")
    parser.add_argument("--test_dataset", type=str, help="Path to test dataset (Arrow format)")
    parser.add_argument("--test_jsonl", type=str, help="Path to test cases JSONL file")
    parser.add_argument("--agent", type=str, help="Evaluate specific agent")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle test dataset before evaluation")
    parser.add_argument("--limit", type=int, help="Limit evaluation to the first N samples")
    parser.add_argument("--snomed_filter", type=str, help="Filter test cases by SNOMED CT prefix")
    parser.add_argument("--base_model", type=str, help="Path to base model")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cpu)")
    parser.add_argument("--agents_dir", type=str, help="Path to agents directory")
    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.test_dataset:
        parser.error("The --test_dataset argument (path to JSONL file) is required.")

    if args.agent:
        logger.info(f"Received request to evaluate single agent: {args.agent}")
    else:
        logger.info("No specific agent requested via --agent flag. Attempting Router mode.")
        if not Router:
             parser.error("Router mode requested (no --agent specified), but Router class could not be imported. Cannot proceed.")

    # --- Print Arguments Before Call --- >
    print(f"[DEBUG] Calling Evaluator with: base_model='{args.base_model}', target_agent='{args.agent}', device='{args.device}', agents_dir='{args.agents_dir}'")
    # < --- End Print Arguments ---

    # --- Initialize and Run Evaluator ---
    try:
        # CORRECTED instantiation: Pass all relevant args
        evaluator = Evaluator(
            base_model_path=args.base_model,
            target_agent=args.agent, # Ensure target_agent is passed
            device=args.device,
            agents_dir=args.agents_dir # Ensure agents_dir is passed
        )

        evaluator.evaluate_dataset(
            dataset_path=args.test_dataset,
            snomed_filter=args.snomed_filter,
            limit=args.limit
        )
        logger.info("Evaluation finished successfully.")

    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
         logger.error(f"Evaluation failed during initialization or execution: {e}", exc_info=False) # Show traceback only if needed
         # Print simplified error message for common issues
         if isinstance(e, FileNotFoundError):
              logger.error(f"Please ensure the path exists: {e}")
         elif isinstance(e, ImportError):
              logger.error(f"A required library or module is missing or could not be imported: {e}")
         sys.exit(1)
    except Exception as e:
         logger.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
         sys.exit(1)


if __name__ == "__main__":
    main() 