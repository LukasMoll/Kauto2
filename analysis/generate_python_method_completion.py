import argparse
import csv
import json

import torch
from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
torch.set_default_device("cuda")


parser = argparse.ArgumentParser(description='Script to generate kotlin code for line completion based on eval data')

# Add two string arguments
parser.add_argument('model_id', type=str, help='model id')
parser.add_argument('fine_tuned_model_id', type=str, help='fine tuned model id')
args = parser.parse_args()


model_id = args.model_name
fine_tuned_model_id = args.fine_tuned_model_name


# Correctly initialize the PeftModel
# model_id = "microsoft/Phi-3-mini-4k-instruct"
# fine_tuned_model_path = "LukasM1/phi-3_kotlin_fine_tuned_4k2"
adapter_name = "personal_copilot"

# Load model and tokenizer from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load the model with correct configuration setup
# Make sure that the configuration being passed does not include unexpected arguments.
fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, fine_tuned_model_id, adapter_name=adapter_name)

# Additional operations
fine_tuned_model.add_weighted_adapter([adapter_name], [0.8], "best_personal_copilot")
fine_tuned_model.set_adapter("best_personal_copilot")

def batch_inference(batch_inputs, tokenizer, model, max_length):
    # Encode all inputs in the batch
    inputs = tokenizer(batch_inputs, padding=True, return_tensors="pt", truncation=True)
    # Perform inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs

def get_method_body(code):
    lines = code.splitlines()
    method_body = []
    current_indent = None

    for line in lines:
        if current_indent is None:
            # Determine the initial indentation on the first line of the method body
            current_indent = len(line) - len(line.lstrip())
            method_body.append(line.strip())
        else:
            # Continue capturing lines that are part of the method body
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= current_indent:
                # If indentation decreases or remains the same, stop capturing
                break
            method_body.append(line.strip())

    return " ".join(method_body).strip()



result = []
with open('test.jsonl') as f:
    for line in f:
        result.append(json.loads(line))
for i, input in enumerate(result[:100]):
    prompt = input['signature'] + "\n\"\"\"" + input['docstring'] + "\"\"\"\n"
    prompt = tokenizer.decode(tokenizer.encode(prompt, padding=True,
                              return_tensors="pt", truncation=True)[0])
    inputs = tokenizer(prompt, max_length=3000, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200 + len(inputs[0]))
    outputs_fine_tuned = fine_tuned_model.generate(**inputs, max_length=200 + len(inputs[0]))
    generated = "\n".join(tokenizer.batch_decode(outputs)[0].splitlines()[prompt.count("\n"):])
    generated_fine_tuned = "\n".join(tokenizer.batch_decode(outputs_fine_tuned)[0].splitlines()[prompt.count("\n"):])
    method_body = get_method_body(generated)
    method_body_fine_tuned = get_method_body(generated_fine_tuned)
    with open('normal.txt', 'a') as file:
        file.write(method_body + '\n')
    with open('fine_tuned.txt', 'a') as file:
        file.write(method_body_fine_tuned + '\n')
    print(f"{i+1}/100")
