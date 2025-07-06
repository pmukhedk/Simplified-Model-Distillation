from pathlib import Path
from datasets import load_dataset
from model_evaluation import evaluate_model
from models_wrapper import load_summarizer  # or load_sentiment_pipeline
from model_summerization import plot_all_metrics_vs_dataset_size
import pandas as pd
import gc

def run_and_cleanup(skill_type, model_name, inputs, references, dataset_size, output_csv):
    summarizer = load_summarizer(model_name)
    evaluate_model(
        skill_type= skill_type,
        trained_model=summarizer,
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        output_csv=output_csv
    )
    # cleanup
    del summarizer
    gc.collect()
    gc.collect()
    gc.collect()


if __name__ == "__main__":
    dataset_size = 300
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{dataset_size}]")

    inputs = dataset["article"]
    references = dataset["highlights"] #This is the human summerization

    file_name= 'withrouge_recall_calculations_al_weights.csv'
    model_name = "ooor/t5-small-distilled-summarization"

    run_and_cleanup(
        skill_type='summarization',
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        output_csv=file_name
    )

'''ooor/t5-small-distilled-summarization
    model_name = "t5-small"
    run_and_cleanup(
        skill_type='summarization',
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        output_csv=file_name
    )

    model_name = "eprasad/t5-small-llama70b-distill-summarization"
    run_and_cleanup(
        skill_type='summarization',
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        output_csv=file_name
    )
    model_name = "eprasad/t5-small-qwen3-distill-summarization"
    run_and_cleanup(
        skill_type='summarization',
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        output_csv=file_name
    )
    model_name = "AhilanPonnusamy/distilled-t5small-summarizer"
    run_and_cleanup(
        skill_type='summarization',
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        output_csv=file_name
    )'''

