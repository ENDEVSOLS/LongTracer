"""
Claim Splitter - Simple sentence splitting with meta-statement detection.
"""

import re
from typing import List


HONEST_UNCERTAINTY_PATTERNS = [
    r"(?i)the provided documents do not contain",
    r"(?i)the documents do not contain",
    r"(?i)I don't have information",
    r"(?i)I cannot find information",
    r"(?i)the context does not provide",
    r"(?i)there is no information about",
    r"(?i)not mentioned in the",
    r"(?i)no specific information",
    r"(?i)I could not find",
]

HALLUCINATION_PATTERNS = [
    r"(?i)based on (?:my |general )?knowledge",
    r"(?i)from my training",
    r"(?i)I know that",
    r"(?i)generally speaking",
    r"(?i)it is well known",
    r"(?i)as we all know",
    r"(?i)in my understanding",
]


def is_meta_statement(text: str) -> bool:
    """Check if text is an honest uncertainty statement."""
    for pattern in HONEST_UNCERTAINTY_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def is_hallucination_pattern(text: str) -> bool:
    """Check if text contains hallucination patterns."""
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def split_into_claims(text: str) -> List[str]:
    """
    Split text into individual claims (sentences).

    Args:
        text: LLM response text.

    Returns:
        List of claim strings.
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    if len(text) <= 10:
        return []

    protected = text
    protected = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', protected)
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr|i\.e|e\.g)\.\s', r'\1<ABBR> ', protected)

    sentences = re.split(r'(?<=[.!?])\s+', protected)

    claims = []
    for s in sentences:
        s = s.replace('<DECIMAL>', '.')
        s = s.replace('<ABBR>', '.')
        s = s.strip()
        if len(s) > 15:
            claims.append(s)

    return claims


def analyze_claim(claim: str) -> dict:
    """Analyze a claim for meta-statements and hallucination patterns."""
    return {
        "is_meta_statement": is_meta_statement(claim),
        "has_hallucination_pattern": is_hallucination_pattern(claim)
    }
