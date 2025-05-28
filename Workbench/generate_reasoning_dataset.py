import csv
import sys
import os

def truncate_to_words(text, max_words=100):
    """Truncate text to the first max_words words."""
    words = text.split()
    return ' '.join(words[:max_words])

def generate_reasoning_dataset(filename):
    # Validate file existence
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found.")
        return
    
    # Generate new file name
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_new.csv"
    
    with open(filename, 'r', encoding='utf-8') as infile, open(new_filename, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header row
        writer.writerow(['text', 'label'])
        
        for row in reader:
            if len(row) < 4:
                print(f"Skipping row (not enough columns): {row}")
                continue
            
            # Truncate review and reasoning
            review_truncated = truncate_to_words(row[1].strip(), max_words=100)
            reasoning_truncated = truncate_to_words(row[2].strip(), max_words=100)
            
            # Format text with tags
            review = f"Review: {review_truncated}."
            reasoning = f" Reasoning: {reasoning_truncated}."
            sentiment = row[3].strip()
            final_sentiment = f" Final Sentiment: {sentiment}"
            
            text = review + reasoning + final_sentiment
            writer.writerow([text, sentiment])
    
    print(f"Generated file: {new_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_reasoning_dataset.py <filename>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    generate_reasoning_dataset(input_file)
