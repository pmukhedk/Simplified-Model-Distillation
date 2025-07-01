import evaluate
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
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

    # Compute per-example scores to access recall
    rouge_f1_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False,
        use_stemmer=True
    )

    rouge1_f1 = rouge_f1_scores['rouge1']
    rougeL_f1 = rouge_f1_scores['rougeL']
    # Aggregate recall
    avg_rouge1_f1 = np.mean(rouge1_f1)
    avg_rougeL_f1 = np.mean(rougeL_f1)

    ####For the recalls use the rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1',"rougeL"],use_stemmer=True)
    rouge1_recalls=[]
    rougeL_recalls=[]
    for pred,ref in zip(predictions, references):
        scores = scorer.score(ref,pred)
        rouge1_recalls.append(scores['rouge1'].recall)
        rougeL_recalls.append(scores['rougeL'].recall)

    # Aggregate recall
    avg_rouge1_recall = np.mean(rouge1_recalls)
    avg_rougeL_recall = np.mean(rougeL_recalls)

    # BERTScore
    bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_recall = np.mean(bert_result['recall'])
    bert_f1 = np.mean(bert_result['f1'])

    cosine_sim = np.mean([
        compute_cosine_similarity(p, r) for p, r in zip(predictions, references)
    ])

    return {
        'ROUGE-1-F1': avg_rouge1_f1,
        'ROUGE-L-F1': avg_rougeL_f1,
        'ROUGE-1-Recall': avg_rouge1_recall,
        'ROUGE-L-Recall': avg_rougeL_recall,
        'BERTScore-Recall': bert_recall,
        'BERTScore-F1': bert_f1,
        'CosineSimilarity': cosine_sim
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
