# Workbench for the dataset prep and training

1. This folder contains all the scripts used for preparing the dataset with knowledge distillation from the Teacher models. The steps will give you a general guidance. You may need to tweak and fix some environment dependent issues along the way.
   
2. The prepared datasets (```train.csv and valid.csv```) are available under the respective dataset folders **cnn_dailymail & yelp-review-dataset**. All the baseline dataset along with the intermediate data files are available under the backup folders.
   
3. Let us start by fine-tuning the Sentiment Analysis base model.
  * Make sure you have the distillation-venv activated and all the dependencies are installed (created earlier)
  * Download smollm 135M model (https://huggingface.co/HuggingFaceTB/SmolLM-135M) to ./smollm-135m folder using ```huggingface-cli download HuggingFaceTB/SmolLM-135M --local-dir ./smollm-135``` command.
  * Fine-tune the student model (with distilled knowledge from teacher) using the command below
```bash
    autotrain text-classification --train \
  --project-name sentiment-distillation-smollm \
  --data-path ./yelp-review-dataset \
  --train-split train \
  --valid-split valid \
  --model ./smollm-135M \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 5 \
  --mixed-precision bf16 \
  --optimizer adamw_torch \
  --scheduler linear \
  --max-grad-norm 1 \
  --max-seq-length 256 \
  --warmup-ratio 0.1 \
  --weight-decay 0 \
  --seed 42 \
  --gradient-accumulation 1 \
  --eval-strategy epoch \
  --early-stopping-patience 5 \
  --early-stopping-threshold 0.01 \
  --save-total-limit 1 \
  --logging-steps 10 \
  --text-column text \
  --target-column label
```
>[!NOTE]
>If you are stuck with pad token error use the ```patch_tokenizer.py``` script under the **Workbench** folder to fix it.

  * After some time (40 mins approximately) the fine-tuning should be completed and the new distilled model should be available at ```sentiment-distillation-smollm``` folder (change the CHECKPOINT_DIR and TOKENIZER_DIR to point to your environment as needed).

  * You may now test the distiled model by updating the following code in ```sentiment_distilled_model_test.py```
```bash
CHECKPOINT_DIR = "./sentiment-distillation-smollm/checkpoint-xxx" 
TOKENIZER_DIR = "./sentiment-distillation-smollm"  # or CHECKPOINT_DIR if tokenizer saved there

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
#tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
```
4. Let us fine-tune summarization base model next
  * Make sure you have the distillation-venv active and all the dependencies are installed (created earlier)
  * Download t5-small model (https://huggingface.co/google-t5/t5-small) to ./t5-small folder using ```huggingface-cli download``` command.
  * Fine-tune the student model (with distilled knowledge from teacher) using the script below
```bash
python summarization_training.py
```
  * After fine-tuning the new distilled model should be available at summarization-distillation-t5small folder (change the CHECKPOINT_DIR and TOKENIZER_DIR to point to your environment as needed).

  * You may now test the distiled model by updating the following code in ```summary_distilled_model_test.py```
```bash
CHECKPOINT_DIR = "./summarization-distillation-t5small/checkpoint-xxx" 
TOKENIZER_DIR = "./summarization-distillation-t5small"  # or CHECKPOINT_DIR if tokenizer saved there

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
#tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
```
5. Upload the distlled models to your huggingface repo for future reference.

## Generating the Knowledge Distillation Dataset Using Teacher Models

6. Utilize the implementation details provided in the ```teacher_and_student_sentiment_analysis.py``` and ```teacher_summarization_distillation_dataprep.py``` scripts to generate and capture reasoning(rationale) and label for the fine-tuning dataset.
   
7. Use the generated datasets to create training and validation splits for Sentiment Analysis and Summarization. For Sentiment Analysis, run ```generate_reasoning_dataset.py```; for Summarization, run ```generate_summarization_reasoning_dataset.py```. These scripts produce the raw datasets, which you should then split into train.csv and valid.csv files. Move these files to their respective dataset folders (yelp-review-dataset or cnn_dailymail). For example,```python generate_summarization_reasoning_dataset.py ./cnn_dailymail/backup/train_with_generated_summary.csv```
  * **Recommendation**:

     - Allocate approximately 10–15% of the data for the validation set.
     - Optionally, reserve the last 50 or 100 rows for use in post-distillation verification scripts.

8. Execute the fine-tuning scripts explained in steps 3 and 4 above to create distilled Sentiment Analysis and Summarization SLMs.
### 🎉 Have Fun! Extend, explore, and enjoy! 😄
