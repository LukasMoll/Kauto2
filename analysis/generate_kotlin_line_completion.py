import csv
import json

import pandas as pd
import torch
from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel

import random

torch.set_default_device("cuda")

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load model and tokenizer from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Correctly initialize the PeftModel
fine_tuned_model_path = "../fine_tuning/phi-3_kotlin_fine_tuned"
adapter_name = "personal_copilot"

# Load the model with correct configuration setup
# Make sure that the configuration being passed does not include unexpected arguments.
fine_tuned_model = PeftModel.from_pretrained(model, fine_tuned_model_path, adapter_name=adapter_name)

# Additional operations
fine_tuned_model.add_weighted_adapter(["personal_copilot"], [0.8], "best_personal_copilot")
fine_tuned_model.set_adapter("best_personal_copilot")


random.seed(10)


df = pd.read_parquet("../dataset_parser/train_data/kotlin_train_data.parquet")
file_name = "kotlin_line_completion_results.csv"
with open(file_name, "w") as file:
    file.write("actual_next_line,predicted_next_line,similarity,"
               "exact_match,predicted_next_line_fine_tuned,similarity_fine_tuned,exact_match_fine_tuned\n")


def predict_next_line(upper_half, model2):
    inputs = tokenizer(upper_half, max_length=3800, truncation=True, return_tensors="pt", return_attention_mask=False)
    input_length = len(inputs[0])
    outputs = model2.generate(**inputs, max_length=input_length + 50)
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
upper_halve_to_next_line = []
for kotlin_code in df["text"]:
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
    exact_match = actual_next_line == predicted_next_line
    exact_match_fine_tuned = actual_next_line == predicted_next_line_fine_tuned
    with open(file_name, "a") as file:
        file.write(f"{actual_next_line},{predicted_next_line},{similarity},{exact_match},"
                   f"{predicted_next_line_fine_tuned},{similarity_fine_tuned},{exact_match_fine_tuned}\n")
    exact_matches += exact_match
    exact_matches_fine_tuned += exact_match_fine_tuned
    similarity_list.append(similarity)
    similarity_list_fine_tuned.append(similarity_fine_tuned)
    print(f"Exact matches            : {exact_matches}")
    print(f"Exact matches fine tuned : {exact_matches_fine_tuned}")
    print(f"Similarity               : {round(sum(similarity_list) / len(similarity_list), 2)}")
    print(f"Similarity fine tuned    : {round(sum(similarity_list_fine_tuned) / len(similarity_list_fine_tuned), 2)}")
