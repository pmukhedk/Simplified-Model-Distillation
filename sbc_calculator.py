import nltk
import numpy as np
from typing import List, Dict
import re

# Setup
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords


def extract_keywords(text: str) -> set:
    stop_words = set(stopwords.words("english"))
    words = re.findall(r"\b\w+\b", text.lower())
    return set([word for word in words if word not in stop_words])

def compute_completeness(reference: str, prediction: str) -> float:
    ref_keywords = extract_keywords(reference)
    pred_keywords = extract_keywords(prediction)
    union = ref_keywords.union(pred_keywords)
    if not union:
        return 0.0
    missing = ref_keywords - pred_keywords
    extra = pred_keywords - ref_keywords
    return 1 - (len(missing) + len(extra)) / len(union)


def compute_sbc_scores(bertF1,cosine_values,predictions, references) -> Dict[str, float]:
    semantic_scores = bertF1
    # Completeness
    completeness_scores = []
    for ref,pred in zip(references,predictions):
        completeness_scores.append(compute_completeness(ref, pred))

    cosine_scores=cosine_values
    # Combine all into SBC
    # Combine all into SBC
    sbc_scores = [
        (0.7 * semantic) + (0.2 * complete) + (0.1 * cosine)
        for semantic, complete, cosine in zip(semantic_scores, completeness_scores, cosine_values)
    ]

    return {
        "avg_sbc_score": float(np.mean(sbc_scores)),
        "avg_semantic_score": float(np.mean(semantic_scores)),
        "avg_completeness_score": float(np.mean(completeness_scores)),
        "avg_cosine_score": float(np.mean(cosine_scores)),
    }
    
