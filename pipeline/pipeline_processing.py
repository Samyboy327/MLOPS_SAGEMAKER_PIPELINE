import os
import shutil


input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

input_file = os.path.join(input_dir, "customer_clean.csv")
output_file = os.path.join(output_dir, "customer_clean.csv")

os.makedirs(output_dir, exist_ok=True)

print(f"Reading input: {input_file}")

shutil.copy2(input_file, output_file)

print(f"Processed dataset written to: {output_file}")
