import os
import torch
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Paths
model_name_or_path = "./t5-small"
data_dir = "./cnn_dailymail"
output_dir = "./summarization-distillation-t5small"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

print("Loading dataset...")
data_files = {
    "train": os.path.join(data_dir, "train.csv"),
    "validation": os.path.join(data_dir, "valid.csv")
}
dataset = load_dataset("csv", data_files=data_files)

def is_valid(example):
    valid = (
        example.get("text") and example.get("label") and
        isinstance(example["text"], str) and isinstance(example["label"], str) and
        example["text"].strip() != "" and example["label"].strip() != ""
    )
    if not valid:
        print(f"Invalid example filtered out: {example}")
    return valid

print("Filtering invalid dataset examples...")
dataset["train"] = dataset["train"].filter(is_valid)
dataset["validation"] = dataset["validation"].filter(is_valid)

max_input_length = 512
max_target_length = 512
pad_token_id = tokenizer.pad_token_id

def preprocess_function(examples):
    print(f"Preprocessing batch of {len(examples['text'])} examples...")
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["label"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )["input_ids"]

    # Replace pad tokens in labels with -100
    labels = [
        [(l if l != pad_token_id else -100) for l in label_seq]
        for label_seq in labels
    ]

    # Sanity checks to catch empty sequences
    for i, (input_id_seq, label_seq) in enumerate(zip(model_inputs["input_ids"], labels)):
        if not input_id_seq or not label_seq:
            print(f"Warning: Empty sequence in preprocess at index {i}")
        assert isinstance(input_id_seq, list) and len(input_id_seq) > 0, f"Empty input_ids at index {i}"
        assert isinstance(label_seq, list) and len(label_seq) > 0, f"Empty labels at index {i}"

    model_inputs["labels"] = labels
    return model_inputs

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])

def filter_zero_length_examples(example):
    input_ids = example.get("input_ids", None)
    labels = example.get("labels", None)

    if input_ids is None or labels is None:
        print(f"Filtered example missing input_ids or labels: {example}")
        return False

    if not isinstance(input_ids, list) or not isinstance(labels, list):
        print(f"Filtered example with non-list input_ids or labels: input_ids type={type(input_ids)}, labels type={type(labels)}")
        return False

    if len(input_ids) == 0 or len(labels) == 0:
        print("Filtered example with zero-length input_ids or labels")
        return False

    if all(label == -100 for label in labels):
        print("Filtered example with all -100 labels (padding only)")
        return False

    if any((not isinstance(token, int)) for token in input_ids):
        print("Filtered example with non-int tokens in input_ids")
        return False

    if any((not isinstance(token, int)) for token in labels):
        print("Filtered example with non-int tokens in labels")
        return False

    return True

print("Filtering zero-length tokenized examples...")
before_train_len = len(tokenized_datasets["train"])
before_val_len = len(tokenized_datasets["validation"])

tokenized_datasets["train"] = tokenized_datasets["train"].filter(filter_zero_length_examples)
tokenized_datasets["validation"] = tokenized_datasets["validation"].filter(filter_zero_length_examples)

print(f"Filtered train examples: {before_train_len} -> {len(tokenized_datasets['train'])}")
print(f"Filtered validation examples: {before_val_len} -> {len(tokenized_datasets['validation'])}")

print("Setting up data collator...")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print("Loading ROUGE metric...")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Just multiply by 100, no .mid.fmeasure
    return {k: v * 100 for k, v in result.items()}


bf16_flag = False
if torch.cuda.is_available():
    bf16_flag = True
elif torch.backends.mps.is_available():
    bf16_flag = False

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    gradient_accumulation_steps=1,
    predict_with_generate=True,
    logging_steps=10,
    save_strategy="epoch",
    bf16=bf16_flag,
    seed=42,
    report_to="none",
    generation_max_length=max_target_length,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
    greater_is_better=True,
)

print("Sanity check: manual batch from data loader...")
from torch.utils.data import DataLoader
dl = DataLoader(tokenized_datasets["train"], batch_size=4, collate_fn=data_collator)
try:
    batch = next(iter(dl))
    print("Manual batch collated successfully.")
except Exception as e:
    print(f"Error collating manual batch: {e}")
    raise

print("Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01),
    ],
)

print("Starting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    raise

print("Saving model and tokenizer...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
