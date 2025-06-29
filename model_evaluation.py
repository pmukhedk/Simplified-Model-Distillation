import evaluate
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Global sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_cosine_similarity(a, b):
    a_emb = embedding_model.encode(a, convert_to_tensor=True)
    b_emb = embedding_model.encode(b, convert_to_tensor=True)
    return 1 - cosine(a_emb.cpu().numpy(), b_emb.cpu().numpy())


def evaluate_model(skill_type, trained_model, model_name, inputs, references, output_csv='model_metrics.csv'):
    results = {'model_name': model_name, 'skill_type': skill_type}
    predictions = [trained_model(x) for x in inputs]

    if skill_type.lower() == 'summarization':
        results.update(evaluate_summarization(predictions, references))

    elif skill_type.lower() == 'sentiment':
        results.update(evaluate_sentiment(predictions, references))

    elif skill_type.lower() == 'search':
        results.update(evaluate_search(predictions, references))

    else:
        raise ValueError(f"Unknown skill_type: {skill_type}")

    write_results_to_csv(results, output_csv)
    return results


def evaluate_summarization(predictions, references):
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {
        'ROUGE-1': rouge_result['rouge1'],
        'ROUGE-L': rouge_result['rougeL'],
        'BERTScore-F1': np.mean(bert_result['f1']),
        'CosineSimilarity': np.mean([
            compute_cosine_similarity(p, r) for p, r in zip(predictions, references)
        ])
    }


def evaluate_sentiment(predictions, references):
    result = {
        'Accuracy': accuracy_score(references, predictions),
        'F1-Score': f1_score(references, predictions, average='weighted'),
        'Recall': recall_score(references, predictions, average='weighted'),
        'ConfusionMatrix': str(confusion_matrix(references, predictions).tolist()),
        'CosineSimilarity': np.mean([
            compute_cosine_similarity(p, r) for p, r in zip(predictions, references)
        ])
    }

    try:
        if all(isinstance(p, str) and isinstance(r, str) for p, r in zip(predictions, references)):
            bertscore = evaluate.load("bertscore")
            bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
            result['BERTScore-F1'] = np.mean(bert_result['f1'])
        else:
            result['BERTScore-F1'] = 'N/A'
    except Exception as e:
        result['BERTScore-F1'] = f"Error: {str(e)}"

    return result


def evaluate_search(predictions, references):
    def pairwise_kl(p, r):
        p = np.array(p) + 1e-9
        r = np.array(r) + 1e-9
        return np.sum(p * np.log(p / r))

    def top_k_agreement(p, r, k=5):
        return len(set(p[:k]).intersection(set(r[:k]))) / k

    return {
        'PairwiseKL': np.mean([pairwise_kl(p, r) for p, r in zip(predictions, references)]),
        'TopKAgreement@5': np.mean([top_k_agreement(p, r, k=5) for p, r in zip(predictions, references)]),
        'CosineSimilarity': np.mean([
            compute_cosine_similarity(" ".join(map(str, p)), " ".join(map(str, r)))
            for p, r in zip(predictions, references)
        ])
    }


def write_results_to_csv(results, output_csv):
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(results)
