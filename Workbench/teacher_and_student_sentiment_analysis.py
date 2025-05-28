import csv
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# CONFIGS
INPUT_FILE = './yelp-review-dataset/test.csv'
OUTPUT_FILE = './yelp-review-dataset/test_with_sentiment_PRE_TRAIN_SLM.csv'
#NUM_ROWS = 15000
NUM_ROWS = 100
#TEACHER_MODEL = 'gemma3:12b'
TEACHER_MODEL = 'smollm:135m'
NUM_WORKERS = 5  # Adjust based on your system
BATCH_SIZE = 10  # Write to file every 10 rows

PROMPT_TEMPLATE = """
You are an expert at sentiment analysis.
Classify the following Yelp review as positive, negative, or neutral. Also provide a brief explanation for your reasoning.
Review: "{review}"
Respond in the format:
Reasoning: <reasoning text>
Sentiment: <positive/negative/neutral>
"""

def call_teacher_llm(review_id, review_text):
    prompt = PROMPT_TEMPLATE.format(review=review_text)
    try:
        response = ollama.chat(model=TEACHER_MODEL, messages=[{"role": "user", "content": prompt}])
        reasoning, sentiment = parse_response(response['message']['content'])
        return review_id, review_text, reasoning, sentiment
    except Exception as e:
        print(f"Error with Ollama API for review {review_id}: {e}")
        return review_id, review_text, "Error", "Error"

def parse_response(response_text):
    reasoning = ""
    sentiment = ""
    lines = response_text.strip().splitlines()
    for line in lines:
        if line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
        if line.lower().startswith("sentiment:"):
            sentiment = line.split(":", 1)[1].strip().lower()
    return reasoning, sentiment

def process_yelp_reviews():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        rows = []
        for i, row in enumerate(reader):
            if i >= NUM_ROWS:
                break
            if len(row) < 2:
                continue
            rows.append((row[0], row[1]))  # (review_id, review_text)

        print(f"🚀 Starting with {len(rows)} reviews using {NUM_WORKERS} threads...")

        batch_rows = []
        completed = 0

        try:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(call_teacher_llm, r[0], r[1]) for r in rows]

                for future in as_completed(futures):
                    result = future.result()
                    batch_rows.append(result)
                    completed += 1

                    print(f"[{completed}/{len(rows)}] Processed: {result[1][:60]}...")

                    if len(batch_rows) >= BATCH_SIZE:
                        writer.writerows(batch_rows)
                        batch_rows = []
                        outfile.flush()  # Ensure file is up-to-date

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted! Saving progress...")

        # Write remaining rows
        if batch_rows:
            writer.writerows(batch_rows)
            outfile.flush()

if __name__ == "__main__":
    start_time = time.time()
    process_yelp_reviews()
    elapsed = time.time() - start_time
    print(f"\n Processing complete in {elapsed:.2f} seconds! Output saved to {OUTPUT_FILE}")

