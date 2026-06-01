import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Load the model and tokenizer
# Point to the directory where your model files are located in the container
MODEL_PATH = "./model"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# this runs locally on Mac but not containerised
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH, 
#     torch_dtype=torch.float16, # Use float16 to save memory
#     device_map="auto"          # Automatically uses GPU if available
# )

# Change device_map="auto" to "cpu" - this is for conterised environment where GPU is not available. Note that this will be slower than using GPU.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float32, # float32 is better for CPU
    device_map="cpu"
)

print("Model loaded successfully.")

# 3. Define the input data structure
class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

# 4. Define the inference endpoint
@app.post("/generate")
def generate(request: InferenceRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=request.max_new_tokens
            )
            
        result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return {"generated_text": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this:
# uvicorn inference:app --host 0.0.0.0 --port 8000