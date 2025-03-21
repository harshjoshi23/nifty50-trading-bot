
import kagglehub
import shutil
import os

# Download latest dataset
path = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")
print("Path to dataset files:", path)

# dir 
dest_dir = "../data/"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Move all CSV files to the data/ directory
for file in os.listdir(path):
    if file.endswith(".csv"):
        src_path = os.path.join(path, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(src_path, dest_path)
        print(f"Moved {file} to {dest_dir}")

# List the files in the data directory
print("Files in data/ directory:", os.listdir(dest_dir))