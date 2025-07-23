import evaluate
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
#from sbc_calculator_ahilan import compute_sbc_scores
#from sbc_calculator import compute_sbc_scores
from sbc_calculator_lemm import compute_sbc_scores
from sentence_transformers import util
from transformers import pipeline

# Global sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def write_per_sample_results(model_name, inputs, references, predictions):
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    filename = f"results_{safe_model_name}.csv"
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['text', 'reference', 'prediction'])
        writer.writeheader()
        for inp, ref, pred in zip(inputs, references, predictions):
            writer.writerow({
                'text': inp.strip(),
                'reference': ref.strip(),
                'prediction': pred.strip()
            })

def compute_cosine_similarity(a, b):
    emb1 = embedding_model.encode(a, convert_to_tensor=True)
    emb2 = embedding_model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def compute_sample_agreement(predictions, references):
    return sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)) / len(predictions)

def get_model_size_mb(model):
    try:
        return round(sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2), 2)
    except:
        return "N/A"

def evaluate_model(skill_type, trained_model, model_name, inputs, references,dataset_size, prompt_template, output_csv='model_metrics.csv'):
    results = {
                'model_name': model_name,
                'skill_type': skill_type,
                'dataset_size': dataset_size,
                'prompt_template': prompt_template
            }
    predictions = [trained_model(prompt_template.format(text=x)) for x in inputs]

    write_per_sample_results(model_name,inputs,references,predictions)

    model_size =  get_model_size_mb(trained_model)
    if skill_type.lower() == 'summarization':
        results.update(evaluate_summarization(predictions, references, model_size))

    elif skill_type.lower() == 'sentiment':
        results.update(evaluate_sentiment(predictions, references))

    elif skill_type.lower() == 'search':
        results.update(evaluate_search(predictions, references))

    else:
        raise ValueError(f"Unknown skill_type: {skill_type}")

    write_results_to_csv(results, output_csv)
    return results


def evaluate_summarization(predictions, references,model_size):
    bertscore = evaluate.load("bertscore")
    ####For the recalls use the rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1',"rougeL"],use_stemmer=True)
    rouge1_recalls=[]
    rougeL_recalls=[]

    rouge1_f1=[]
    rougeL_f1=[]

    for pred,ref in zip(predictions, references):
        scores = scorer.score(ref,pred)
        rouge1_recalls.append(scores['rouge1'].recall)
        rougeL_recalls.append(scores['rougeL'].recall)
        rouge1_f1.append(scores['rouge1'].fmeasure)
        rougeL_f1.append(scores['rougeL'].fmeasure)

    # Aggregate recall
    avg_rouge1_recall = np.mean(rouge1_recalls)
    avg_rougeL_recall = np.mean(rougeL_recalls)
    avg_rouge1_f1 = np.mean(rouge1_f1)
    avg_rougeL_f1 = np.mean(rougeL_f1)

    # BERTScore
    bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_recall = np.mean(bert_result['recall'])
    bert_f1 = np.mean(bert_result['f1'])
    cosine_values=[compute_cosine_similarity(p, r) for p, r in zip(predictions, references)]

    cosine_sim = np.mean(cosine_values)
    sbc_value= compute_sbc_scores(bert_result['f1'],cosine_values,predictions, references)


    return {
        'ROUGE-1-F1': round(avg_rouge1_f1,4),
        'ROUGE-L-F1': round(avg_rougeL_f1,4),
        'ROUGE-1-Recall': round(avg_rouge1_recall,4),
        'ROUGE-L-Recall': round(avg_rougeL_recall,4),
        'BERTScore-Recall': round(bert_recall,4),
        'BERTScore-F1': round(bert_f1,4),
        'CosineSimilarity': round(cosine_sim,4),
        'SBC-Score': round(sbc_value.get("avg_sbc_score"),4),
        'SBC-SemanticScore': round(sbc_value.get("avg_semantic_score"),4),
        'SBC-CompletenessScore': round(sbc_value.get("avg_completeness_score"),4),
        'SBC-CosineScore': round(sbc_value.get("avg_cosine_score"),4)
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
