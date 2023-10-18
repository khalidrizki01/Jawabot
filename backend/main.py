from fastapi import FastAPI, Request
import uvicorn
import os
from dotenv import load_dotenv
from agent import JawabotAgent
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

origins = [
    "http://frontend:5173",
    "http://jawabot-dev.web.id"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods=['*'],
    allow_headers=['*'],
)

jawabot = JawabotAgent()

@app.get("/")
def read_root():
    return {"message": "Selamat datang di Jawabot API"}

@app.post("/ask")
async def answer(request: Request):
    try:
        req = await request.json()
        question = req['question']
        # return {"message":question}
        return {"message":jawabot.async_generate(question)}
    except Exception as e:
        return {"error", str(e)}
    
# if __name__ == "__main__" :
#     uvicorn.run(app, host = 'localhost', port = 8000)