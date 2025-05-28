from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./smollm-135M")
model = AutoModelForCausalLM.from_pretrained("./smollm-135M")

# Set the pad_token to eos_token if pad_token is not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("./smollm-135M")

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id
    model.save_pretrained("./smollm-135M")

