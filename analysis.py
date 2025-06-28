import evaluate
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from scipy.spatial.distance import cosine
from transformers import pipeline

from sentence_transformers import SentenceTransformer

# Load once globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_cosine_similarity(a, b):
    a_emb = embedding_model.encode(a, convert_to_tensor=True)
    b_emb = embedding_model.encode(b, convert_to_tensor=True)
    return 1 - cosine(a_emb.cpu().numpy(), b_emb.cpu().numpy())

def evaluate_model(skill_type, trained_model, model_name, inputs, references, output_csv='model_metrics2.csv'):
    results = {'model_name': model_name, 'skill_type': skill_type}
    predictions = [trained_model(x) for x in inputs]

    if skill_type.lower() == 'summarization':
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")
        rouge_result = rouge.compute(predictions=predictions, references=references)
        bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")

        results.update({
            'ROUGE-1': rouge_result['rouge1'],
            'ROUGE-L': rouge_result['rougeL'],
            'BERTScore-F1': np.mean(bert_result['f1']),
            'CosineSimilarity': np.mean([compute_cosine_similarity(p, r) for p, r in zip(predictions, references)])
        })

    elif skill_type.lower() == 'sentiment':
        results.update({
            'Accuracy': accuracy_score(references, predictions),
            'F1-Score': f1_score(references, predictions, average='weighted'),
            'Recall': recall_score(references, predictions, average='weighted'),
            'ConfusionMatrix': str(confusion_matrix(references, predictions).tolist()),
            'CosineSimilarity': np.mean([compute_cosine_similarity(p, r) for p, r in zip(predictions, references)])
        })
        try:
            if all(isinstance(p, str) and isinstance(r, str) for p, r in zip(predictions, references)):
                bertscore = evaluate.load("bertscore")
                bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
                results['BERTScore-F1'] = np.mean(bert_result['f1'])
            else:
                results['BERTScore-F1'] = 'N/A'
        except Exception as e:
            results['BERTScore-F1'] = f"Error: {str(e)}"

    elif skill_type.lower() == 'search':
        def pairwise_kl(p, r):
            p = np.array(p) + 1e-9
            r = np.array(r) + 1e-9
            return np.sum(p * np.log(p / r))

        def top_k_agreement(p, r, k=5):
            return len(set(p[:k]).intersection(set(r[:k]))) / k

        results.update({
            'PairwiseKL': np.mean([pairwise_kl(p, r) for p, r in zip(predictions, references)]),
            'TopKAgreement@5': np.mean([top_k_agreement(p, r, k=5) for p, r in zip(predictions, references)]),
            'CosineSimilarity': np.mean([compute_cosine_similarity(" ".join(map(str, p)), " ".join(map(str, r)))
                                         for p, r in zip(predictions, references)])
        })

    else:
        raise ValueError(f"Unknown skill_type: {skill_type}")

    # Write to CSV
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

    return results
'''
model_name="eprasad/distilled-clm-200m"
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LEDTokenizer, LEDForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
#tokenizer = BartTokenizer.from_pretrained(model_name)
#model = BartForConditionalGeneration.from_pretrained(model_name)
#tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#model = GPT2LMHeadModel.from_pretrained(model_name)



from transformers import GPT2Tokenizer, GPT2LMHeadModel
model_name="eprasad/distilled-t5-small"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name="t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def summarizer_wrapper(input_text):
    input_ids = tokenizer("summarize: " + input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

'''
model_name="eprasad/sentiment-distillation-smollm"
sentiment_pipeline = pipeline("sentiment-analysis",model_name)

def sentiment_wrapper(input_text):
    result = sentiment_pipeline(input_text)[0]
    return result['label'].lower()  # e.g., 'positive' or 'negative'
'''

from datasets import load_dataset
ds = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")  # small sample
inputs = ds["article"]
references = ds["highlights"]

evaluate_model(
    skill_type='summarization',
    trained_model=summarizer_wrapper,
    model_name=model_name,
    inputs=inputs,
    references=references,
    output_csv="100sample_results.csv"
)

'''
#evaluate_model(
    skill_type='summarization',
    trained_model=summarizer_wrapper,
    model_name=model_name,
    inputs=["The Eiffel Tower is one of the most iconic landmarks in Paris."],
    references=["The Eiffel Tower is a famous landmark in Paris."]
)'''


'''
eprasad/sentiment-distillation-smollm
evaluate_model(
    skill_type='sentiment',
    trained_model=sentiment_wrapper,
    model_name='AhilanPonnusamy/distilled-t5small-summarizer',
    inputs=["I love this product", "It was terrible"],
    references=["positive", "negative"]
)


eprasad/sentiment-distillation-smollm
'''
