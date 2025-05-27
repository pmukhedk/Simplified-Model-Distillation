import csv

# File paths
INPUT_FILE = './ResultsSentimentAnalysis.csv'
OUTPUT_FILE = './Results_with_different_outputs.csv'

# Initialize counters
pre_train_total = 0
pre_train_match = 0
post_train_total = 0
post_train_match = 0
distillation_total = 0
distillation_match = 0

# Prepare a list for post-training misaligned rows
misaligned_rows = []

# Read and process the CSV
with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Capture the header

    for row in reader:
        if len(row) < 4:
            continue  # Skip incomplete rows

        col2 = row[1].strip().lower()  # Column 2
        col3 = row[2].strip().lower()  # Column 3
        col4 = row[3].strip().lower()  # Column 4
        #col5 = row[4].strip().lower()

        # Pre-training accuracy (col2 vs col3)
        pre_train_total += 1
        if col2 == col3:
            pre_train_match += 1


        distillation_total += 1
        if col2 == col4:
            distillation_match += 1
        else:
            misaligned_rows.append(row)

# Write misaligned rows to the output file
with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)  # Write the header row
    writer.writerows(misaligned_rows)  # Write all misaligned rows

# Calculate percentages
pre_train_accuracy = (pre_train_match / pre_train_total) * 100 if pre_train_total else 0
distillation_accuracy = (distillation_match / distillation_total) * 100 if distillation_total else 0

# Display results
print(f"✅ Pre-training Relative Accuracy: {pre_train_accuracy:.2f}%")
print(f"✅ Distillation Relative Accuracy: {distillation_accuracy:.2f}%")
print(f"🔎 Misaligned rows saved to {OUTPUT_FILE}")
