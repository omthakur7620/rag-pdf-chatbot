# app/utils.py

from typing import List


def calculate_confidence(scores: List[float]) -> float:
    """
    Calculate confidence score based on similarity scores.

    Args:
        scores (List[float]): Similarity scores from vector search

    Returns:
        float: Confidence score between 0 and 1
    """

    if not scores:
        return 0.0

    # Average similarity score
    avg_score = sum(scores) / len(scores)

    # Clamp between 0 and 1 (safety)
    confidence = max(0.0, min(1.0, avg_score))

    return round(confidence, 2)
