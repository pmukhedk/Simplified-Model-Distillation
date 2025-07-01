from datasets import load_dataset
from model_evaluation import evaluate_model
from models_wrapper import load_summarizer  # or load_sentiment_pipeline

if __name__ == "__main__":
    model_name = "t5-small"
    summarizer = load_summarizer(model_name)

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
    inputs = dataset["article"]
    references = dataset["highlights"]

    evaluate_model(
        skill_type='summarization',
        trained_model=summarizer,
        model_name=model_name,
        inputs=inputs,
        references=references,
        output_csv="withrouge_recall_calculations.csv"
    )

    model_name = "eprasad/t5-small-llama70b-distill-summarization"
    summarizer = load_summarizer(model_name)
    evaluate_model(
        skill_type='summarization',
        trained_model=summarizer,
        model_name=model_name,
        inputs=inputs,
        references=references,
        output_csv="withrouge_recall_calculations.csv"
    )
    model_name = "eprasad/t5-small-qwen3-distill-summarization"
    summarizer = load_summarizer(model_name)
    evaluate_model(
        skill_type='summarization',
        trained_model=summarizer,
        model_name=model_name,
        inputs=inputs,
        references=references,
        output_csv="withrouge_recall_calculations.csv"
    )
    model_name = "AhilanPonnusamy/distilled-t5small-summarizer"
    summarizer = load_summarizer(model_name)
    evaluate_model(
        skill_type='summarization',
        trained_model=summarizer,
        model_name=model_name,
        inputs=inputs,
        references=references,
        output_csv="withrouge_recall_calculations.csv"
    )


