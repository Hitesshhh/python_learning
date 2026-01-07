from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

app = FastAPI()


load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # React
    ],
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, DELETE
    allow_headers=["*"],   # Authorization, Content-Type
)


def generate_stream(prompt: str):
    print(prompt)
    stream = client.chat.completions.create(
        model="zai-org/GLM-4.7:novita",
        messages=[
            {
                "role": "system",
                "content": "You are a professional Python tutor. Explain concepts clearly with examples."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

@app.post("/chat")
async def chat(request: Request):
    print(request.body())
    body = await request.json()
    prompt = body.get("prompt")

    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/plain"
    )
