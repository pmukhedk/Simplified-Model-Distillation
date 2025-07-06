import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq # Import Groq

# CONFIGS
INPUT_FILE = './cnn_dailymail/train.csv'
OUTPUT_FILE = './cnn_dailymail/train_with_generated_summary_deepseek-r1-distill-llama-70b.csv'
START_ROW = 1 # Skip header
END_ROW = 3000
NUM_WORKERS = 5
BATCH_SIZE = 10
TEACHER_MODEL = "deepseek-r1-distill-llama-70b" # Changed to a Groq-compatible model

# Initialize Groq client (replace with your actual API key)
# It's recommended to load the API key from environment variables for security.
# Example: groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_client = Groq()

PROMPT_TEMPLATE = """
You are an expert at summarization.
Read the following text and generate a concise summary. Also, explain your reasoning briefly.
Text: "{text}"
Respond in the format:
Reasoning: <reasoning text>
Summary: <summary text>
"""

def parse_response(response_text):
    reasoning = ""
    summary = ""
    lines = response_text.strip().splitlines()
    for line in lines:
        if line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
    return reasoning, summary

def call_teacher_llm(row_id, text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=TEACHER_MODEL,
            temperature=0.7, # You can adjust temperature if needed
            max_tokens=500, # You can adjust max_tokens if needed
        )
        content = chat_completion.choices[0].message.content
        reasoning, summary = parse_response(content)
        return row_id, text, reasoning, summary
    except Exception as e:
        print(f"❌ Error with Groq API for row {row_id}: {e}")
        return row_id, text, "Error", "Error"

def process_csv():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        rows = []
        for i, row in enumerate(reader):
            if i < START_ROW:
                continue
            if i >= END_ROW:
                break
            if len(row) < 3:
                continue
            rows.append((row[0], row[1], row[2]))  # (id, text, existing_summary)

        print(f"🚀 Processing {len(rows)} rows using {NUM_WORKERS} threads...")

        batch_rows = []
        completed = 0

        try:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # We need to create a list of futures and map them back to original row data
                futures = {executor.submit(call_teacher_llm, row_id, text): (row_id, text, existing_summary) for row_id, text, existing_summary in rows}

                for future in as_completed(futures):
                    original_row_data = futures[future]
                    row_id, text, existing_summary = original_row_data

                    result = future.result()
                    # result is (row_id, text, reasoning, summary) from call_teacher_llm
                    # We need to use the original text and existing_summary for the output row
                    batch_rows.append([row_id, text, existing_summary, result[2], result[3]])
                    completed += 1

                    print(f"[{completed}/{len(rows)}] Processed row {row_id}...")

                    if len(batch_rows) >= BATCH_SIZE:
                        writer.writerows(batch_rows)
                        batch_rows = []
                        outfile.flush()

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted! Saving progress...")

        if batch_rows:
            writer.writerows(batch_rows)
            outfile.flush()

if __name__ == "__main__":
    start_time = time.time()
    process_csv()
    elapsed = time.time() - start_time
    print(f"\n✅ Processing complete in {elapsed:.2f} seconds! Output saved to {OUTPUT_FILE}")