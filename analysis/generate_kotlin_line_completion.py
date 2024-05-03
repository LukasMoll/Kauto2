import argparse
import csv
import json

import pandas as pd
import torch
from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel

import random


parser = argparse.ArgumentParser(description='Script to generate kotlin code for line completion based on eval data')

# Add two string arguments
parser.add_argument('model_id', type=str, help='model id')
parser.add_argument('fine_tuned_model_id', type=str, help='fine tuned model id')
args = parser.parse_args()


model_id = args.model_id
fine_tuned_model_id = args.fine_tuned_model_id

torch.set_default_device("cuda")

# Load model and tokenizer from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Correctly initialize the PeftModel
fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
adapter_name = "personal_copilot"

# Load the model with correct configuration setup
# Make sure that the configuration being passed does not include unexpected arguments.
fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, fine_tuned_model_id, adapter_name=adapter_name)

# Additional operations
fine_tuned_model.add_weighted_adapter(["personal_copilot"], [0.8], "best_personal_copilot")
fine_tuned_model.set_adapter("best_personal_copilot")


random.seed(10)


df = pd.read_parquet("../dataset_parser/eval_data/kotlin_eval_data2.parquet")
file_name = "kotlin_line_completion_results.csv"
with open(file_name, "w") as file:
    file.write("actual_next_line,predicted_next_line,similarity,"
               "exact_match,predicted_next_line_fine_tuned,similarity_fine_tuned,exact_match_fine_tuned\n")


def predict_next_line(upper_half, model2):
    inputs = tokenizer(upper_half, max_length=1900, truncation=True, return_tensors="pt", return_attention_mask=False)
    input_length = len(inputs[0])
    outputs = model2.generate(**inputs, max_length=input_length + 100)
    text = tokenizer.batch_decode(outputs)[0]
    result = text.splitlines()[upper_half.count("\n")]
    if text.count("\n") == upper_half.count("\n"):
        print("not enough tokens to generate new line")
        print(text)
    return result


exact_matches = 0
exact_matches_fine_tuned = 0
similarity_list = []
similarity_list_fine_tuned = []
exact_matches_list = []
exact_matches_list_fine_tuned = []
upper_halve_to_next_line = []
text_len = len(df["text"])
for i, kotlin_code in enumerate(df["text"]):
    lines = kotlin_code.splitlines()
    if len(lines) < 4:
        continue
    split_line = random.randint(1, len(lines) - 2)
    upper_half = "\n".join(lines[max(0, split_line - 50):split_line]) + "\n"
    actual_next_line = ""
    while actual_next_line.strip() == "":
        actual_next_line = lines[split_line]
        split_line += 1
    predicted_next_line = predict_next_line(upper_half, model)
    predicted_next_line_fine_tuned = predict_next_line(upper_half, fine_tuned_model)
    similarity = fuzz.ratio(actual_next_line, predicted_next_line)
    similarity_fine_tuned = fuzz.ratio(actual_next_line, predicted_next_line_fine_tuned)
    exact_matches_list.append(actual_next_line == predicted_next_line)
    exact_matches_list_fine_tuned.append(actual_next_line == predicted_next_line_fine_tuned)
    similarity_list.append(similarity)
    similarity_list_fine_tuned.append(similarity_fine_tuned)
    print(f"{i+1}/{text_len}")
    print(f"Exact matches             : {round(100 * sum(exact_matches_list) / len(exact_matches_list), 2)}%")
    print(f"Exact matches fine tuned  : {round(100 * sum(exact_matches_list_fine_tuned) / len(exact_matches_list_fine_tuned), 2)}%")
    print(f"Similarity                : {round(sum(similarity_list) / len(similarity_list), 2)}%")
    print(f"Similarity fine tuned     : {round(sum(similarity_list_fine_tuned) / len(similarity_list_fine_tuned), 2)}%")
    print(f"Exact matches improvement : {round(100 * (sum(exact_matches_list_fine_tuned) / sum(exact_matches_list) - 1), 2)}%")
    print(f"Similarity improvement    : {round(100 * (sum(similarity_list_fine_tuned) / sum(similarity_list) - 1), 2)}%")
