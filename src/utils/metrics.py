#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.40
Usage:  Imported by evaluation scripts (e.g., src/evaluate.py, src/baselines/logit.py).
        Provides a `DiagnosisMetrics` class and potentially standalone metric functions.

Objective of the Code:
------------
Provides functions and potentially a class for calculating, tracking, and 
reporting performance metrics relevant to the medical diagnosis task. This 
likely includes standard classification metrics (accuracy, precision, recall, F1) 
and potentially utilities for handling confidence scores or latency measurements.
"""

import json
import logging
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_METRICS_DIR = Path("metrics")


class DiagnosisMetrics:
    """
    Class for tracking and calculating metrics for the medical diagnosis system.
    """
    
    def __init__(self, metrics_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the metrics tracker.
        
        Args:
            metrics_dir: Directory to store metrics data (defaults to "metrics")
        """
        self.metrics_dir = Path(metrics_dir) if metrics_dir else DEFAULT_METRICS_DIR
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
        self.diagnosis_records = []
        self.latency_records = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Metrics tracker initialized with session ID: {self.current_session_id}")
        logger.info(f"Metrics will be stored in: {self.metrics_dir}")
    
    def record_diagnosis(self, 
                         patient_id: str,
                         predicted_diagnosis: str,
                         ground_truth: Optional[str] = None,
                         confidence: float = 0.0,
                         model_name: str = "",
                         additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a single diagnosis result for later analysis.
        
        Args:
            patient_id: Unique identifier for the patient
            predicted_diagnosis: The diagnosis predicted by the system
            ground_truth: The correct diagnosis (if known)
            confidence: Confidence score for the prediction
            model_name: Name of the model used for diagnosis
            additional_info: Any additional information to record
        """
        if additional_info is None:
            additional_info = {}
            
        record = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "model_name": model_name,
            "predicted_diagnosis": predicted_diagnosis,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "correct": (ground_truth == predicted_diagnosis) if ground_truth else None,
            **additional_info
        }
        
        self.diagnosis_records.append(record)
        logger.debug(f"Recorded diagnosis for patient {patient_id}")
    
    def record_latency(self, 
                      operation: str, 
                      duration_ms: float,
                      success: bool = True,
                      additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Record latency for an operation.
        
        Args:
            operation: Name of the operation (e.g., "routing", "diagnosis")
            duration_ms: Duration in milliseconds
            success: Whether the operation was successful
            additional_info: Any additional information to record
        """
        if additional_info is None:
            additional_info = {}
            
        record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **additional_info
        }
        
        self.latency_records.append(record)
        logger.debug(f"Recorded latency for {operation}: {duration_ms}ms")
    
    def calculate_accuracy(self) -> float:
        """
        Calculate accuracy of diagnoses where ground truth is available.
        
        Returns:
            Accuracy as a float between 0 and 1
        """
        records_with_ground_truth = [r for r in self.diagnosis_records if r["ground_truth"] is not None]
        
        if not records_with_ground_truth:
            logger.warning("No records with ground truth available to calculate accuracy")
            return 0.0
        
        correct_count = sum(1 for r in records_with_ground_truth if r["correct"])
        accuracy = correct_count / len(records_with_ground_truth)
        
        logger.info(f"Calculated accuracy: {accuracy:.4f} from {len(records_with_ground_truth)} records")
        return accuracy
    
    def calculate_precision_recall_f1(self, 
                                     condition: str) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for a specific condition.
        
        Args:
            condition: The condition/diagnosis to calculate metrics for
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        records_with_ground_truth = [r for r in self.diagnosis_records if r["ground_truth"] is not None]
        
        if not records_with_ground_truth:
            logger.warning("No records with ground truth available")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        true_positives = sum(1 for r in records_with_ground_truth 
                           if r["predicted_diagnosis"] == condition and r["ground_truth"] == condition)
        
        false_positives = sum(1 for r in records_with_ground_truth 
                            if r["predicted_diagnosis"] == condition and r["ground_truth"] != condition)
        
        false_negatives = sum(1 for r in records_with_ground_truth 
                            if r["predicted_diagnosis"] != condition and r["ground_truth"] == condition)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        logger.info(f"Calculated metrics for condition '{condition}': {metrics}")
        return metrics
    
    def calculate_confusion_matrix(self, 
                                  conditions: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Calculate confusion matrix for specified conditions.
        
        Args:
            conditions: List of conditions to include in the confusion matrix
            
        Returns:
            Nested dictionary representing the confusion matrix
        """
        records_with_ground_truth = [r for r in self.diagnosis_records if r["ground_truth"] is not None]
        
        if not records_with_ground_truth:
            logger.warning("No records with ground truth available")
            return {}
        
        # Initialize confusion matrix
        confusion_matrix = {actual: {predicted: 0 for predicted in conditions} 
                          for actual in conditions}
        
        # Fill confusion matrix
        for record in records_with_ground_truth:
            actual = record["ground_truth"]
            predicted = record["predicted_diagnosis"]
            
            if actual in conditions and predicted in conditions:
                confusion_matrix[actual][predicted] += 1
        
        logger.info(f"Generated confusion matrix for {len(conditions)} conditions")
        return confusion_matrix
    
    def calculate_average_latency(self, operation: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate average latency statistics.
        
        Args:
            operation: Optional operation name to filter by
            
        Returns:
            Dictionary with average, min, max, and p95 latencies
        """
        if operation:
            relevant_records = [r for r in self.latency_records if r["operation"] == operation]
        else:
            relevant_records = self.latency_records
        
        if not relevant_records:
            logger.warning(f"No latency records found for {operation if operation else 'any operation'}")
            return {
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p95_ms": 0.0,
                "count": 0
            }
        
        durations = [r["duration_ms"] for r in relevant_records]
        
        stats = {
            "avg_ms": np.mean(durations),
            "min_ms": np.min(durations),
            "max_ms": np.max(durations),
            "p95_ms": np.percentile(durations, 95),
            "count": len(durations)
        }
        
        operation_str = f" for {operation}" if operation else ""
        logger.info(f"Calculated latency statistics{operation_str}: avg={stats['avg_ms']:.2f}ms, "
                   f"p95={stats['p95_ms']:.2f}ms, count={stats['count']}")
        
        return stats
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save all metrics to a JSON file.
        
        Args:
            filename: Optional filename to use
            
        Returns:
            Path to the saved metrics file
        """
        if not filename:
            filename = f"metrics_{self.current_session_id}.json"
        
        filepath = self.metrics_dir / filename
        
        metrics = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_diagnoses": len(self.diagnosis_records),
                "total_operations": len(self.latency_records),
                "accuracy": self.calculate_accuracy(),
                "latency": self.calculate_average_latency()
            },
            "diagnosis_records": self.diagnosis_records,
            "latency_records": self.latency_records
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
        return str(filepath)
    
    def generate_report(self, conditions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            conditions: Optional list of conditions to include in detailed metrics
            
        Returns:
            Dictionary with report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "overall": {
                "total_diagnoses": len(self.diagnosis_records),
                "accuracy": self.calculate_accuracy(),
                "latency": self.calculate_average_latency()
            },
            "operations": {},
            "conditions": {}
        }
        
        # Get unique operations
        operations = set(r["operation"] for r in self.latency_records)
        for operation in operations:
            report["operations"][operation] = self.calculate_average_latency(operation)
        
        # Get metrics for specific conditions if provided
        if conditions:
            for condition in conditions:
                report["conditions"][condition] = self.calculate_precision_recall_f1(condition)
            
            report["confusion_matrix"] = self.calculate_confusion_matrix(conditions)
        
        logger.info(f"Generated performance report with {len(report['operations'])} operations "
                   f"and {len(report.get('conditions', {}))} conditions")
        
        return report


def time_operation(func):
    """
    Decorator to measure execution time of functions.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that measures execution time
    """
    def wrapper(*args, **kwargs):
        metrics = None
        # Try to find the metrics object in args or kwargs
        for arg in args:
            if isinstance(arg, DiagnosisMetrics):
                metrics = arg
                break
        
        if metrics is None and 'metrics' in kwargs:
            metrics = kwargs['metrics']
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # If metrics object is available, record the latency
            if metrics is not None:
                operation = func.__name__
                metrics.record_latency(operation, duration_ms, success)
            
            logger.debug(f"Function {func.__name__} took {duration_ms:.2f}ms")
        
        return result
    
    return wrapper


if __name__ == "__main__":
    # Test the metrics module
    metrics = DiagnosisMetrics()
    
    # Record some test diagnoses
    metrics.record_diagnosis(
        patient_id="P001",
        predicted_diagnosis="Hypertension",
        ground_truth="Hypertension",
        confidence=0.85,
        model_name="CardiovascularAgent"
    )
    
    metrics.record_diagnosis(
        patient_id="P002",
        predicted_diagnosis="Type 2 Diabetes",
        ground_truth="Type 2 Diabetes",
        confidence=0.92,
        model_name="EndocrineAgent"
    )
    
    metrics.record_diagnosis(
        patient_id="P003",
        predicted_diagnosis="Asthma",
        ground_truth="COPD",
        confidence=0.67,
        model_name="RespiratoryAgent"
    )
    
    # Record some latencies
    metrics.record_latency("routing", 120.5)
    metrics.record_latency("diagnosis", 850.2)
    metrics.record_latency("diagnosis", 780.9)
    
    # Test accuracy calculation
    accuracy = metrics.calculate_accuracy()
    print(f"Accuracy: {accuracy:.4f}")
    
    # Test precision, recall, F1 calculation
    condition_metrics = metrics.calculate_precision_recall_f1("Hypertension")
    print(f"Hypertension metrics: {condition_metrics}")
    
    # Test confusion matrix
    conditions = ["Hypertension", "Type 2 Diabetes", "COPD", "Asthma"]
    confusion_matrix = metrics.calculate_confusion_matrix(conditions)
    print("Confusion Matrix:")
    for actual in conditions:
        row = [confusion_matrix.get(actual, {}).get(predicted, 0) for predicted in conditions]
        print(f"{actual}: {row}")
    
    # Test latency statistics
    latency_stats = metrics.calculate_average_latency("diagnosis")
    print(f"Diagnosis latency stats: {latency_stats}")
    
    # Test saving metrics
    filepath = metrics.save_metrics("test_metrics.json")
    print(f"Metrics saved to: {filepath}")
    
    # Test generating report
    report = metrics.generate_report(conditions)
    print("Report generated with sections:", list(report.keys()))
    
    # Test the decorator
    @time_operation
    def test_function(x, y, metrics=None):
        time.sleep(0.1)  # Simulate work
        return x + y
    
    result = test_function(5, 3, metrics=metrics)
    print(f"Test function result: {result}")
    
    # Print final latency records
    print(f"Number of latency records: {len(metrics.latency_records)}")
    print(f"Last latency record: {metrics.latency_records[-1]}") 