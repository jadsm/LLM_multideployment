# TxGemma-2B-Predict

TxGemma-2B-Predict is a specialized, open-weights large language model (LLM) developed by Google DeepMind. It is a fine-tuned version of the Gemma-2 model specifically optimized for therapeutic development and drug discovery. Unlike general-purpose models, this variant is designed to synthesize scientific data to help researchers predict chemical properties and analyze therapeutic targets.

## Installation

First create a python environment using your favourite env management system. 

Then, install hf cli sdk: 
`pip install -U huggingface_hub`

To download the model (you need to login first via `hf auth login`, and copy token from https://huggingface.co/settings/tokens): 
`hf download google/txgemma-2b-predict`

Rename the folder to model: 
`mv txgemma-2b-predict model`

To install the necessary dependencies, run the following command in your terminal:
`pip install -r requirements.txt --upgrade`

To load the model in memory and run locally, run: 
`python local_inference.py`

To serve the model as an API locally, run: 
`uvicorn inference:app --host 0.0.0.0 --port 8000`

To build the container locally, run:
`docker build -t gemma-tx .`

To run the container locally run:
`docker run --gpus all -p 8000:8000 gemma-tx`

To quick deploy onto cloud run from code directly:
`gcloud run deploy gemma-tx --source . --region europe-west2 --port 8000 --allow-unauthenticated --cpu-boost --timeout=300s --memory=8Gi --cpu=4`

Alternatively, you can do this the manual way, by creating a repository (model registry) and then importing the image - then deploy the container in cloud run.

To deploy the image to Google Cloud Run, run: 
```
gcloud run deploy my-fastapi-service \
  --image us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-app:v1 \
  --region us-central1
```

To call the local or remote API, run (from the CLI):
``` 
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your input here", "max_new_tokens": 50}'
```
Note that the input needs to be of a specific format. Examples for this can be found in local_inference.py and remote_inference.py.

Or (from python):
`python remote_inference.py`

## Key Components Explained
model-0000X-of-00003.safetensors: These are the actual model weights split into chunks. The transformers library reads these automatically using the model.safetensors.index.json file.

tokenizer.json / tokenizer_config.json: These are essential for converting your text into the numerical format (tokens) the model understands.

config.json: This defines the architecture of the model (number of layers, heads, hidden size, etc.).

device_map="auto": This is highly recommended. It detects if you have an NVIDIA GPU (via CUDA) and will offload the model to it. If you don't have enough VRAM, it will intelligently use your system RAM (CPU) for the remaining parts.

## What is it used for?
TxGemma-2B-Predict is intended to act as an efficient "generalist" in the scientific space. It helps researchers with:

Predict Properties: Determine characteristics of small molecules, proteins, or nucleic acids (e.g., solubility, toxicity, or blood-brain barrier permeability).

Analyze Therapeutic Data: Process scientific contexts to assist in drug candidate prioritization.

Bridge the Gap: It serves as a middle ground between highly specific, single-task models and general, non-specialized LLMs, offering better performance on scientific benchmarks while maintaining conversational reasoning.

## How to format your inputs
Because this model is trained on scientific data from the Therapeutics Data Commons (TDC), it performs best when inputs are structured in a specific way. It does not just respond to general "chat"; it expects a formal task structure.

Your file tdc_prompts.json contains templates that you should use to format your requests. A typical prompt should include:

Instructions: Clear commands (e.g., "Answer the following question about drug properties.")

Context: The scientific background or information needed to solve the problem.

Question: The specific prediction or inquiry.

Input Data: The technical data, such as:

SMILES strings: Representing chemical structures (e.g., CN1C(=O)CN=C(...)).

Amino acid/Nucleotide sequences: For proteins or DNA/RNA analysis.

Natural Language: Descriptions of diseases, cell lines, or biological mechanisms.