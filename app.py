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
        f"You are a precise data-extraction tool. Look carefully at the image and answer this question: {req.question}\n\n"
        "Rules:\n"
        "- Output ONLY the raw answer value, nothing else.\n"
        "- No explanations, no labels, no full sentences.\n"
        "- No currency symbols, no percent signs, no commas in numbers, no units.\n"
        "- If the answer is a number, output just the number (e.g. 4089.35).\n"
        "- If the answer is a category/text label, output just that label exactly as written in the image.\n"
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
        }],
        temperature=0
    )
    raw = completion.choices[0].message.content.strip()

    # Clean up common junk the model might add
    cleaned = raw.strip().strip('"').strip("'")
    # Remove trailing periods
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    # If model added a prefix like "Answer: X", strip it
    if ":" in cleaned and len(cleaned.split(":")[0]) < 15:
        cleaned = cleaned.split(":", 1)[-1].strip()

    return {"answer": cleaned}

@app.get("/")
def root():
    return {"status": "ok"}
