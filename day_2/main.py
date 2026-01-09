from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os


load_dotenv()

app = FastAPI()

class Body(BaseModel):
    prompt:str


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

file = client.files.create(
    file=open("training.jsonl", "rb"),
    purpose="fine-tune"
)

job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

job = client.fine_tuning.jobs.retrieve(job.id)
print(job.status)

@app.get('/')
async def check_server():
    return {"success":"true","message":"python backend is running !"}

@app.post('/generate')
async def generate(body:Body):
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {
                "role": "system",
                "content": "You are a helpful chatbot, friendly and human-like."
            },
            {
                "role": "user",
                "content": body.prompt
            }
        ],
        temperature=0.5,
        max_output_tokens=256
    )

    return {
        "success": True,
        "output": response.output_text
    }