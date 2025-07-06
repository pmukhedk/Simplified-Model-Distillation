import nltk
import numpy as np
from typing import List, Dict
import re
import spacy
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu

# Setup
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords


# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(req1, req2):
    emb1 = model.encode(req1, convert_to_tensor=True)
    emb2 = model.encode(req2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# BLEU Score
def bleu_score(req1, req2):
    return sentence_bleu([req1.split()], req2.split())

def extract_keywords(text: str) -> set:
    doc = nlp(text)
    return {token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "PROPN"]}

def compute_completeness(reference: str, prediction: str) -> float:
    keywords1 = extract_keywords(reference)
    keywords2 = extract_keywords(prediction)
    missing = keywords1 - keywords2
    extra = keywords2 - keywords1
    # Calculate penalty based on number of missing and extra keywords
    penalty = len(missing) + len(extra)
    # We now normalize based on the total number of unique keywords across both texts
    total_keywords = len(keywords1.union(keywords2))
    score = max(0, 1 - (penalty / max(total_keywords, 1)))  # Avoid division by zero
    return score



def compute_sbc_scores(bertF1,cosine_values,predictions, references) -> Dict[str, float]:
    # Completeness
    completeness_scores = []
    semantic_scores =[]
    bleu_scores =[]
    sbc_scores=[]
    for ref,pred in zip(references,predictions):
        completeness_score = compute_completeness(ref, pred)
        completeness_scores.append(completeness_score)
        bleu_score_value = bleu_score(ref, pred)
        bleu_scores.append(bleu_score_value)
        semantic_score = semantic_similarity(ref, pred)
        semantic_scores.append(semantic_score)
        sbc_scores.append((0.7 * semantic_score) + (0.1 * bleu_score_value) + (0.2 * completeness_score))

    return {
        "avg_sbc_score": float(np.mean(sbc_scores)),
        "avg_semantic_score": float(np.mean(semantic_scores)),
        "avg_completeness_score": float(np.mean(completeness_scores)),
        "avg_cosine_score": float(np.mean(bleu_scores)),
    }
