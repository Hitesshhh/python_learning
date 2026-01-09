from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

class Body(BaseModel):
    prompt: str


model_id = "mistralai/Mistral-7B-v0.1"  
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True, 
     device_map={"": "cpu"}, 
    llm_int8_enable_fp32_cpu_offload=True  
)


@app.get("/")
async def check_server():
    return {"success": True, "message": "Local LLM backend is running!"}


@app.post("/generate")
async def generate(body: Body):
    # Tokenize prompt
    inputs = tokenizer(body.prompt, return_tensors="pt").to(model.device)

    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  
        do_sample=True,       
        temperature=0.7
    )


    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"success": True, "output": text}
