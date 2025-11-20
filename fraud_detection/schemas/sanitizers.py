import math
import numpy as np

def sanitize_value(value):
    """Fix NaN, Infinity, numpy types, and convert to JSON-safe values."""

    # None stays None
    if value is None:
        return None

    # Convert numpy types to native types
    if isinstance(value, (np.generic,)):
        value = value.item()

    # Convert datetime to ISO string
    if hasattr(value, "isoformat"):
        return value.isoformat()

    # Strings are fine
    if isinstance(value, str):
        return value

    # ints/floats should be checked for validity
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # For unknown types, return string representation
    return str(value)


def sanitize_dict(data: dict):
    """Recursively sanitize dictionary for JSON serialization."""
    if not isinstance(data, dict):
        return data

    sanitized = {}

    for key, value in data.items():

        if isinstance(value, dict):
            sanitized[key] = sanitize_dict(value)

        elif isinstance(value, list):
            sanitized[key] = [sanitize_value(v) for v in value]

        else:
            sanitized[key] = sanitize_value(value)

    return sanitized
