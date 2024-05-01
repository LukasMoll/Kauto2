import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

# Set up the device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
BATCH_SIZE = 32  # You can adjust this depending on your GPU memory

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to predict next lines in batch
def batch_predict_next_lines(codes, model):
    # Prepare the batch
    batch = tokenizer(codes, padding=True, truncation=True, return_tensors="pt", max_length=2000)
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model.generate(**batch, max_length=2100, pad_token_id=tokenizer.eos_token_id)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Assume we want the first generated line after the input
    next_lines = [pred.splitlines()[1] if len(pred.splitlines()) > 1 else "" for pred in predictions]
    return next_lines

# Function to process a DataFrame in batches
def process_dataframe(df, model, batch_size=BATCH_SIZE):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_codes = df.iloc[i:i + batch_size].tolist()
            futures.append(executor.submit(batch_predict_next_lines, batch_codes, model))
        for future in futures:
            results.extend(future.result())
    return results

# Load your data
df = pd.read_parquet("../dataset_parser/eval_data/kotlin_eval_data.parquet")

# Preprocess the data to get the upper half of each code snippet as input
df['input'] = df['text'].apply(lambda x: '\n'.join(x.splitlines()[:len(x.splitlines())//2]))

# Predict next lines in batches
predictions = process_dataframe(df['input'], model)

# Example to merge predictions back to DataFrame for further analysis or saving
df['predicted_next_line'] = predictions

# Output results to a CSV file
df.to_csv('predicted_results.csv', index=False)

print("Processing complete. Results are saved in 'predicted_results.csv'.")
