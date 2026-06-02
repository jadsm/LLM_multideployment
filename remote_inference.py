import json
import requests
import time

# 1. Load the prompt template
with open("tdc_prompts.json", "r") as f:
    tdc_prompts_json = json.load(f)

# 2. Define your inputs
task_name = "BBB_Martins"
input_type = "{Drug SMILES}"
drug_smiles = "CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21"

# 3. Format the prompt
# We use .replace() as you did to swap the placeholder for the actual data
TDC_PROMPT = tdc_prompts_json[task_name].replace(input_type, drug_smiles)

# 4. Prepare the payload for the API
# locally
# url = "http://localhost:8000/generate"
# remotely
url = "https://gemma-tx-915753635870.europe-west2.run.app"
payload = {
    "prompt": TDC_PROMPT,
    "max_new_tokens": 50
}

# 5. Send the POST request
try:
    tic = time.time()

    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raises an error for bad status codes
    
    # 6. Parse and print the result
    result = response.json()
    print("Formatted prompt sent to server:\n")
    print(TDC_PROMPT)
    print("\n--- Model Response ---")
    print(result.get("generated_text"))
    toc = time.time()
    print(f"Tokenization completed in {toc - tic:.2f} seconds")

except requests.exceptions.RequestException as e:
    print(f"An error occurred while connecting to the model: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")