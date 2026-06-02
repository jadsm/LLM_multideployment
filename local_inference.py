from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

# Path to the folder containing the files shown in your screenshot
model_path = "model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model
# device_map="auto" automatically handles splitting the model across GPU/CPU
# Source - https://stackoverflow.com/a/5849861
# Posted by Eli Bendersky, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-01, License - CC BY-SA 4.0

tic = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype="auto"
)
toc = time.time()
print(f"Model loaded in {toc - tic:.2f} seconds")

# download the TDC prompts JSON file from Google Cloud Storage and load it
#gcloud storage cp gs://healthai-us/txgemma/templates/tdc_prompts.json tdc_prompts.json
with open("tdc_prompts.json", "r") as f:
    tdc_prompts_json = json.load(f)

# Example task and input
task_name = "USPTO"#"BBB_Martins"
input_type = "{Product SMILES}"#"{Drug SMILES}"
drug_smiles = "CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21"

TDC_PROMPT = tdc_prompts_json[task_name].replace(input_type, drug_smiles)
print("Formatted prompt:\n")
print(TDC_PROMPT)

# Example Inference
tic = time.time()
inputs = tokenizer(TDC_PROMPT, return_tensors="pt").to(model.device)
toc = time.time()
print(f"Tokenization completed in {toc - tic:.2f} seconds")

tic = time.time()
outputs = model.generate(**inputs, max_new_tokens=50)
toc = time.time()
print(f"Inference completed in {toc - tic:.2f} seconds")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
