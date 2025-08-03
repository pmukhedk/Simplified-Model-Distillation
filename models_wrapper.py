from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    LEDTokenizer, LEDForConditionalGeneration,
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoConfig
)
import torch
import warnings


def get_device():
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) is available. Using MPS backend.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU is available. Using CUDA backend.")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU found. Using CPU. Performance may be slow.")
        return torch.device("cpu")


device = get_device()


def load_summarizer(model_name):
    print(f"\n🔄 Loading summarization model: {model_name}")
    try:
        config = AutoConfig.from_pretrained(model_name)
        print("📦 Model config loaded.")
        print("🔧 Architectures:", config.architectures)
        print("📌 Device in use:", device)
    except Exception as e:
        warnings.warn(f"⚠️ Failed to load model config: {e}")

    if "t5" in model_name.lower():
        print("📥 Downloading T5 tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        print("✅ T5 model loaded successfully.")
        prompt_template = "summarize: {text}"

        def summarizer(input_text):
            #print("📝 Running summarizer on input...")
            encoding = tokenizer(
                prompt_template.format(text=input_text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            output_ids = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_new_tokens=100
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "bart" in model_name.lower():
        print("📥 Downloading BART tokenizer and model...")
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        print("✅ BART model loaded successfully.")
        prompt_template = "{text}"

        def summarizer(input_text):
            #print("📝 Running summarizer on input...")
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "gpt2" in model_name.lower():
        print("📥 Downloading GPT-2 tokenizer and model...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
        model.config.pad_token_id = tokenizer.eos_token_id
        print("✅ GPT-2 model loaded successfully.")
        prompt_template = "Please summarize the following:\n\n{text}\n\nSummary:"

        def summarizer(input_text):
           #print("📝 Running summarizer on input...")
            input_ids = tokenizer.encode(prompt_template.format(text=input_text), return_tensors="pt").to(device)
            output_ids = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "qwen" in model_name.lower():
        print("📥 Downloading Qwen tokenizer and model (remote code enabled)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        qwen_kwargs = {
            "trust_remote_code": True
        }

        if device.type == "cuda":
            print("🚀 Enabling flash attention for CUDA.")
            qwen_kwargs.update({
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
                "device_map": "auto"
            })
        else:
            qwen_kwargs["device_map"] = None  # fallback
            print("⚠️ Flash attention disabled: Not using CUDA.")

        model = AutoModelForCausalLM.from_pretrained(model_name, **qwen_kwargs).eval()
        print("✅ Qwen model loaded successfully.")
        prompt_template = "{text} ignore reasoning and provide only the final summary"

        def summarizer(input_text):
            #print("📝 Running summarizer on input...")
            inputs = tokenizer(
                prompt_template.format(text=input_text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            first_device = next(model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    elif "led" in model_name.lower():
        print("📥 Downloading LED tokenizer and model...")
        tokenizer = LEDTokenizer.from_pretrained(model_name)
        model = LEDForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        print("✅ LED model loaded successfully.")
        prompt_template = "{text}"

        def summarizer(input_text):
            #print("📝 Running summarizer on input...")
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            global_attention_mask = torch.zeros_like(inputs["input_ids"]).to(device)
            global_attention_mask[:, 0] = 1  # Global attention on first token
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=global_attention_mask,
                max_length=100
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    else:
        raise ValueError(f"❌ Unsupported summarization model: {model_name}")

    return summarizer, prompt_template


def load_sentiment_pipeline(model_name="eprasad/sentiment-distillation-smollm"):
    print(f"\n🔄 Loading sentiment model: {model_name}")
    try:
        print("📥 Attempting to use transformers pipeline...")
        sentiment_pipe = pipeline("sentiment-analysis", model=model_name, device=0 if device.type == "cuda" else -1)
        print("✅ Pipeline loaded successfully.")

        def classify_sentiment(text):
            print("🧪 Running sentiment analysis...")
            result = sentiment_pipe(text)[0]
            return result['label'].lower()

        return classify_sentiment

    except Exception as e:
        warnings.warn(f"⚠️ Pipeline load failed, using manual fallback: {e}")
        print("📥 Downloading tokenizer and model manually...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        print("✅ Sentiment model loaded manually.")

        def classify_sentiment(text):
            print("🧪 Running sentiment analysis...")
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=-1).item()
            return model.config.id2label[prediction].lower()

        return classify_sentiment
