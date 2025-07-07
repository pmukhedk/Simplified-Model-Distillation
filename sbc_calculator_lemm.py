import spacy
import numpy as np
from typing import Dict

# Load SpaCy model once (e.g., en_core_web_sm)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Lightweight for speed

def extract_keywords_pos_lemma(text: str) -> set:
    """Extracts lemmatized content words (nouns, verbs, proper nouns) using SpaCy."""
    doc = nlp(text)
    return {token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "VERB", "PROPN"}}

def compute_completeness_pos_lemma(reference: str, prediction: str) -> float:
    ref_keywords = extract_keywords_pos_lemma(reference)
    pred_keywords = extract_keywords_pos_lemma(prediction)
    union = ref_keywords | pred_keywords
    if not union:
        return 0.0
    penalty = len(ref_keywords - pred_keywords) + len(pred_keywords - ref_keywords)
    return max(0.0, 1 - penalty / len(union))

def compute_sbc_scores(bertF1,cosine_values,predictions, references) -> Dict[str, float]:
    semantic_scores = bertF1
    # Completeness
    completeness_scores = []
    for ref,pred in zip(references,predictions):
        completeness_scores.append(compute_completeness_pos_lemma(ref, pred))

    cosine_scores=cosine_values
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

