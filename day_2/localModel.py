from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# device_map 'auto' with CPU offload
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,  # 8-bit quantization
     device_map={"": "cpu"},  # automatically assign layers to GPU/CPU
    llm_int8_enable_fp32_cpu_offload=True  # offload layers to CPU when GPU is full
)

prompt = "Explain FastAPI like I am 5"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Starting generation...")
outputs = model.generate(**inputs, max_new_tokens=50)
print("Done generation!")

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
