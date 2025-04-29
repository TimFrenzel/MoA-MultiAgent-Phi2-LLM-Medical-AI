#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.40
Usage:  Imported by other scripts to validate data structures.
        `is_valid, errors = validate_patient_data(data)`

Objective of the Code:
------------
Provides utility functions for validating and sanitizing patient data 
dictionaries used throughout the MoA pipeline. Ensures required fields 
(like age, gender, symptoms) are present, checks data types and formats, 
and normalizes values (e.g., lowercasing gender). Includes a basic check 
for emergency keywords in symptoms.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_FIELDS = ["age", "gender", "symptoms"]
VALID_GENDERS = ["male", "female", "other", "unknown"]
VALID_ETHNICITIES = ["hispanic", "non-hispanic", "unknown"]
VALID_RACES = ["white", "black", "asian", "native_american", "pacific_islander", "other", "unknown"]

# ICD-10 code pattern (for validating optional input fields that might contain ICD-10)
# Note: The internal system primarily uses SNOMED CT codes derived from Synthea data.
ICD10_PATTERN = re.compile(r'^[A-Z]\d\d(\.\d{1,2})?$')


def validate_patient_data(patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate patient data for completeness and correctness.
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in patient_data:
            errors.append(f"Required field '{field}' is missing")
    
    # Validate age
    if "age" in patient_data:
        try:
            age = int(patient_data["age"])
            if age < 0 or age > 120:
                errors.append(f"Age {age} is outside valid range (0-120)")
        except (ValueError, TypeError):
            errors.append(f"Age '{patient_data['age']}' is not a valid number")
    
    # Validate gender
    if "gender" in patient_data:
        gender = patient_data["gender"].lower()
        if gender not in VALID_GENDERS:
            errors.append(f"Gender '{gender}' is not valid. Must be one of: {', '.join(VALID_GENDERS)}")
    
    # Validate ethnicity if provided
    if "ethnicity" in patient_data:
        ethnicity = patient_data["ethnicity"].lower()
        if ethnicity not in VALID_ETHNICITIES:
            errors.append(f"Ethnicity '{ethnicity}' is not valid. Must be one of: {', '.join(VALID_ETHNICITIES)}")
    
    # Validate race if provided
    if "race" in patient_data:
        race = patient_data["race"].lower()
        if race not in VALID_RACES:
            errors.append(f"Race '{race}' is not valid. Must be one of: {', '.join(VALID_RACES)}")
    
    # Validate symptoms
    if "symptoms" in patient_data:
        symptoms = patient_data["symptoms"]
        if not isinstance(symptoms, list):
            errors.append("Symptoms must be a list")
        elif len(symptoms) == 0:
            errors.append("At least one symptom must be provided")
        else:
            for symptom in symptoms:
                if not isinstance(symptom, str) or len(symptom.strip()) == 0:
                    errors.append(f"Invalid symptom: '{symptom}'")
    
    # Validate specific ICD-10 input codes if provided (Note: Internal domain mapping uses SNOMED CT codes)
    # This validation is for external input; internal codes should be SNOMED CT.
    if "icd10" in patient_data: # Keep 'icd10' key for potential external input format
        icd10_codes_input = patient_data["icd10"]
        if not isinstance(icd10_codes_input, list):
            errors.append("Input field 'icd10' must be a list")
        else:
            for code in icd10_codes_input:
                if not isinstance(code, str) or not ICD10_PATTERN.match(code):
                    errors.append(f"Invalid ICD-10 code format in input field 'icd10': '{code}'")
    
    # Validate date of birth if provided
    if "dob" in patient_data:
        dob = patient_data["dob"]
        try:
            datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid date of birth format: '{dob}'. Use YYYY-MM-DD format.")
    
    # Validate medical history if provided
    if "medical_history" in patient_data:
        medical_history = patient_data["medical_history"]
        if not isinstance(medical_history, list):
            errors.append("Medical history must be a list")
        else:
            for item in medical_history:
                if not isinstance(item, dict):
                    errors.append(f"Medical history item must be a dictionary: {item}")
                elif "condition" not in item:
                    errors.append(f"Medical history item missing 'condition': {item}")
    
    # Validate medications if provided
    if "medications" in patient_data:
        medications = patient_data["medications"]
        if not isinstance(medications, list):
            errors.append("Medications must be a list")
        else:
            for med in medications:
                if not isinstance(med, dict):
                    errors.append(f"Medication must be a dictionary: {med}")
                elif "name" not in med:
                    errors.append(f"Medication missing 'name': {med}")
    
    # Return validation result
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("Patient data validation successful")
    else:
        logger.warning(f"Patient data validation failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"Validation error: {error}")
    
    return is_valid, errors


def sanitize_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize patient data by converting fields to standard formats and removing invalid entries.
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        Sanitized patient data
    """
    sanitized = patient_data.copy()
    
    # Convert age to integer if possible
    if "age" in sanitized:
        try:
            sanitized["age"] = int(sanitized["age"])
        except (ValueError, TypeError):
            sanitized["age"] = None
    
    # Normalize gender
    if "gender" in sanitized:
        gender = sanitized["gender"].lower()
        if gender in VALID_GENDERS:
            sanitized["gender"] = gender
        else:
            sanitized["gender"] = "unknown"
    
    # Normalize ethnicity
    if "ethnicity" in sanitized:
        ethnicity = sanitized["ethnicity"].lower()
        if ethnicity in VALID_ETHNICITIES:
            sanitized["ethnicity"] = ethnicity
        else:
            sanitized["ethnicity"] = "unknown"
    
    # Normalize race
    if "race" in sanitized:
        race = sanitized["race"].lower()
        if race in VALID_RACES:
            sanitized["race"] = race
        else:
            sanitized["race"] = "unknown"
    
    # Sanitize symptoms
    if "symptoms" in sanitized:
        if not isinstance(sanitized["symptoms"], list):
            sanitized["symptoms"] = []
        else:
            sanitized["symptoms"] = [
                symptom.strip().lower() 
                for symptom in sanitized["symptoms"] 
                if isinstance(symptom, str) and symptom.strip()
            ]
    
    # Sanitize specific ICD-10 input codes (Note: Internal domain mapping uses SNOMED CT codes)
    # This sanitation is for external input; internal codes should be SNOMED CT.
    if "icd10" in sanitized: # Keep 'icd10' key for potential external input format
        icd10_codes_input = sanitized["icd10"]
        if isinstance(icd10_codes_input, list):
            valid_codes = []
            for code in icd10_codes_input:
                if isinstance(code, str) and ICD10_PATTERN.match(code):
                    valid_codes.append(code)
            sanitized["icd10"] = valid_codes
        else:
            # If input 'icd10' is not a list, remove it or set to empty list
            sanitized["icd10"] = []
    
    # Sanitize medications
    if "medications" in sanitized:
        if not isinstance(sanitized["medications"], list):
            sanitized["medications"] = []
        else:
            valid_meds = []
            for med in sanitized["medications"]:
                if isinstance(med, dict) and "name" in med:
                    valid_med = {"name": med["name"]}
                    if "dosage" in med:
                        valid_med["dosage"] = med["dosage"]
                    if "frequency" in med:
                        valid_med["frequency"] = med["frequency"]
                    valid_meds.append(valid_med)
            sanitized["medications"] = valid_meds
    
    # Sanitize date of birth if provided
    if "dob" in sanitized:
        dob = sanitized["dob"]
        try:
            datetime.strptime(dob, "%Y-%m-%d")
        except ValueError:
            sanitized["dob"] = None
    
    logger.info("Patient data sanitized successfully")
    return sanitized


def check_for_emergency(patient_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if patient data indicates an emergency situation that requires immediate attention.
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        Tuple of (is_emergency, reason)
    """
    emergency_symptoms = [
        "chest pain", "severe chest pain", "difficulty breathing", 
        "shortness of breath", "severe headache", "sudden confusion",
        "slurred speech", "sudden numbness", "severe abdominal pain",
        "coughing blood", "vomiting blood", "severe bleeding",
        "loss of consciousness", "seizure", "stroke symptoms"
    ]
    
    emergency_reason = None
    is_emergency = False
    
    # Check for emergency symptoms
    if "symptoms" in patient_data and isinstance(patient_data["symptoms"], list):
        patient_symptoms = [s.lower() for s in patient_data["symptoms"] if isinstance(s, str)]
        
        for symptom in emergency_symptoms:
            if symptom in patient_symptoms:
                is_emergency = True
                emergency_reason = f"Emergency symptom detected: {symptom}"
                break
    
    # Check for specific emergency conditions based on potential ICD-10 codes in input
    if "icd10" in patient_data:
        # This part assumes emergency_icd_codes list exists (defined elsewhere or needs adding)
        # emergency_icd_codes = [...] 
        # for code in patient_data["icd10"]:
        #     if code in emergency_icd_codes:
        #         is_emergency = True
        #         emergency_reason = f"Emergency condition indicated by ICD-10 code: {code}"
        #         logger.warning(f"Emergency detected based on ICD-10 code: {code}")
        #         break  # Stop checking once an emergency code is found
        pass # Placeholder: Actual emergency ICD codes need defining

    if is_emergency:
        logger.warning(f"EMERGENCY DETECTED: {emergency_reason}")
    
    return is_emergency, emergency_reason


if __name__ == "__main__":
    # Test the validator
    test_patient_valid = {
        "age": 65,
        "gender": "male",
        "race": "white",
        "ethnicity": "non-hispanic",
        "icd10": ["I25.10", "E11.9"],
        "symptoms": ["chest pain", "shortness of breath", "fatigue"],
        "medications": [
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
            {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"}
        ],
        "medical_history": [
            {"condition": "Type 2 Diabetes", "diagnosed": "2010-05-15"},
            {"condition": "Hypertension", "diagnosed": "2012-03-22"}
        ]
    }
    
    test_patient_invalid = {
        "age": "invalid",
        "gender": "INVALID",
        "symptoms": "headache",  # Should be a list
        "icd10": ["ABC", "123"],  # Invalid ICD-10 codes
    }
    
    test_emergency = {
        "age": 72,
        "gender": "female",
        "symptoms": ["severe chest pain", "shortness of breath", "nausea"],
        "icd10": ["I21.0"]  # Acute myocardial infarction
    }
    
    # Validate test patients
    print("Valid patient validation:")
    is_valid, errors = validate_patient_data(test_patient_valid)
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    print("\nInvalid patient validation:")
    is_valid, errors = validate_patient_data(test_patient_invalid)
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Sanitize invalid patient
    print("\nSanitized patient data:")
    sanitized = sanitize_patient_data(test_patient_invalid)
    print(f"  {sanitized}")
    
    # Check for emergency
    print("\nEmergency check:")
    is_emergency, reason = check_for_emergency(test_emergency)
    print(f"  Emergency: {is_emergency}")
    if reason:
        print(f"  Reason: {reason}") 