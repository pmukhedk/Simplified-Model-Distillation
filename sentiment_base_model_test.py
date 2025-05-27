from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM-135M")

inputs = tokenizer("The movie was amazing!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

label_map = {0: "negative", 1: "neutral", 2: "positive"}
print("Predicted sentiment:", label_map[predicted_class_id])
