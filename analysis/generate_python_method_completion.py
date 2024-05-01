import csv
import json

import torch
from fuzzywuzzy import fuzz
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
torch.set_default_device("cuda")


# Correctly initialize the PeftModel
model_id = "microsoft/Phi-3-mini-4k-instruct"
fine_tuned_path = "../fine_tuning/phi-3_kotlin_fine_tuned"
adapter_name = "personal_copilot"

# Load model and tokenizer from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load the model with correct configuration setup
# Make sure that the configuration being passed does not include unexpected arguments.
fine_tuned_model = PeftModel.from_pretrained(model, fine_tuned_path, adapter_name=adapter_name)

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

def get_method_body(generated: str):
    result = ""
    for i, line in enumerate(generated.splitlines()):
        if line and line[0] != " ":
            break
        if line:
            result += line.strip() + " "
        if line[:10] == "    return" or line[:8] == "  return":
            break
    return result.strip()

result = []
with open('python_test.jsonl') as f:
    for line in f:
        result.append(json.loads(line))
for i, input in enumerate(result[:100]):
    prompt = input['signature'] + "\n\"\"\"" + input['docstring'] + "\"\"\"\n"
    prompt = tokenizer.decode(tokenizer.encode(prompt, padding=True,
                              return_tensors="pt", truncation=True)[0])
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=1024)
    outputs_fine_tuned = fine_tuned_model.generate(**inputs, max_length=1024)
    generated = tokenizer.batch_decode(outputs)[0][len(prompt):]
    generated_fine_tuned = tokenizer.batch_decode(outputs_fine_tuned)[0][len(prompt):]
    method_body = get_method_body(generated)
    method_body_fine_tuned = get_method_body(generated_fine_tuned)
    with open('python_codexglue_results.txt', 'a') as file:
        file.write(method_body + '\n')
    with open('python_codexglue_results_fine_tuned.txt', 'a') as file:
        file.write(method_body_fine_tuned + '\n')
    print(f"{i+1}/100")
