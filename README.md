# Simplified Model Distillation
A lightweight approach for simplifying large language model (LLM) distillation. Fine-tune and distill samller models(SLM) like t5-small and smollm 135M into  efficient models for summarization, sentiment analysis, and more. Built on MacBook Pro with Apple M2 Pro chip and 32 GB Memory.

## Environment Setup

1. Clone the Git Repository

```bash
git clone https://github.com:AhilanPonnusamy/Simplified-Model-Distillation.git
cd Simplified-Model-Distillation

```
2. Create and activate a virtual environement
```bash
python3.12 -m venv distillation-venv
source distillation-venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
## Testing Sentiment Analysis Pre and Post Distillation
4. Run the sentiment analysis test with base model (smollm 135M). Sometimes you may get an unauthorized error even for public model in huggingface, if you face this error login to huggingface using **huggingface-cli** login command.
```bash
python sentiment_base_model_test.py
```
5. Run the sentiment analysis test with the distilled model.
```bash
python sentiment_distilled_model_test.py
```
6. You may notice that the distilled model is more accurate in predicting the sentiment.
   
## Testing Summarization Pre and Post Distillation
7. Run the summarization test with base model (t5-small). Sometimes you may get an unauthorized error even for public model in huggingface, if you face this error login to huggingface using huggingface-cli login command.
```bash
python summary_base_model_test.py
```
8. Run the summarization test with the distilled model.
```bash
python summary_distilled_model_test.py
```
9. You will notice similar results for the base model and distilled model. Please check the **Results** folder to learn more about the improvements the distilled model has attained for summarization task.
### 🎉 Have Fun! Extend, explore, and enjoy! 😄
