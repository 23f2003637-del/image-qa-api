import os, base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
def answer_image(req: Req):
    prompt = (
        f"Look at the image and answer this question: {req.question}\n"
        "Reply with ONLY the raw answer value — no units, no currency symbols, "
        "no extra words. If it's a number, return just the number as a string."
    )
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{req.image_base64}"
                }}
            ]
        }]
    )
    return {"answer": completion.choices[0].message.content.strip()}

@app.get("/")
def root():
    return {"status": "ok"}
