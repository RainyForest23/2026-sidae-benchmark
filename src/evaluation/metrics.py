
import re

def normalize_answer(text):
    """
    Normalize text for comparison.
    """
    return text.strip().lower()

def calculate_accuracy(predictions, references):
    """
    Calculate accuracy.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    
    correct = 0
    for pred, ref in zip(predictions, references):
        if normalize_answer(pred) == normalize_answer(ref):
            correct += 1
            
    return correct / len(predictions) if predictions else 0.0

def extract_option(text):
    """
    Extracts the option letter (A, B, C, D, E) or number (1, 2, 3, 4, 5) from the text.
    First looks for patterns like "(A)", "A.", "Answer: A".
    If not found, takes the first char if it's a valid option.
    """
    # Patterns to look for
    patterns = [
        r"Answer:\s*([A-E1-5])",
        r"Start of Answer:\s*([A-E1-5])",
        r"\(([A-E1-5])\)",
        r"(?i)option\s*([A-E1-5])",
        r"^([A-E1-5])\.",
        r"^([A-E1-5])$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
            
    return text.strip() # Fallback
