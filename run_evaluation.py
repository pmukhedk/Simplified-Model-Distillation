from model_evaluation import evaluate_model
from models_wrapper import load_summarizer  # or load_sentiment_pipeline
from datasets import load_dataset
import gc

def run_and_cleanup(skill_type, model_name, inputs, references, dataset_size, output_csv):
    summarizer,prompt_template = load_summarizer(model_name)
    evaluate_model(
        skill_type= skill_type,
        trained_model=summarizer,
        model_name=model_name,
        inputs=inputs,
        references=references,
        dataset_size=dataset_size,
        prompt_template=prompt_template,
        output_csv=output_csv
    )
    # cleanup
    del summarizer
    gc.collect()
    gc.collect()
    gc.collect()


#

def run_all_models(dataset_size, file_name):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{dataset_size}]")
    inputs = dataset["article"]
    references = dataset["highlights"]
    '''
    model_names = [
        "t5-small",
        "eprasad/t5-small-llama70b-distill-summarization",
        "eprasad/t5-small-qwen3-distill-summarization",
        "AhilanPonnusamy/distilled-t5small-summarizer",
        "ooor/t5-small-distilled-summarization"
    ]'''

    model_names = [
        "t5-small",
        'eprasad/t5-base-summarization-distill-qwen3-32b'
    ]

    for model_name in model_names:
        print(f"🔍 Evaluating {model_name} on dataset_size = {dataset_size}")
        run_and_cleanup(
            skill_type='summarization',
            model_name=model_name,
            inputs=inputs,
            references=references,
            dataset_size=dataset_size,
            output_csv=file_name
        )

if __name__ == "__main__":
    file_name = 'test.csv'

    #for dataset_size in range(100, 1000, 100):  # 100 to 900 inclusive
    for dataset_size in [100]:
        run_all_models(dataset_size, file_name)


