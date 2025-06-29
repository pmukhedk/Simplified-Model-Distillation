from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    LEDTokenizer, LEDForConditionalGeneration,
    pipeline, AutoTokenizer, AutoModelForSequenceClassification
)
import torch
from transformers import AutoConfig


def load_summarizer(model_name):
    config = AutoConfig.from_pretrained(model_name)
    print("Architectures:", config.architectures)
    if "t5" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        def summarizer(input_text):
            input_ids = tokenizer("summarize: " + input_text, return_tensors="pt").input_ids
            output_ids = model.generate(input_ids, max_new_tokens=100)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "bart" in model_name.lower():
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        def summarizer(input_text):
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            output_ids = model.generate(input_ids, max_new_tokens=100)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.eos_token_id
        model.eval()

        def summarizer(input_text):
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            output_ids = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "led" in model_name.lower():
        tokenizer = LEDTokenizer.from_pretrained(model_name)
        model = LEDForConditionalGeneration.from_pretrained(model_name)

        def summarizer(input_text):
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            global_attention_mask = torch.zeros_like(inputs["input_ids"])
            global_attention_mask[:, 0] = 1  # set global attention on [CLS] token
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=global_attention_mask,
                max_length=100
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    else:
        raise ValueError(f"Unsupported summarization model: {model_name}")

    return summarizer


def load_sentiment_pipeline(model_name="eprasad/sentiment-distillation-smollm"):
    # Will attempt to use HF sentiment-analysis pipeline
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model=model_name)

        def classify_sentiment(text):
            result = sentiment_pipe(text)[0]
            return result['label'].lower()

        return classify_sentiment

    except Exception:
        # Fallback for AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        def classify_sentiment(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=-1).item()
            return model.config.id2label[prediction].lower()

        return classify_sentiment
