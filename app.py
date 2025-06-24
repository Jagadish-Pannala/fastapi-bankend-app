# app.py

import uvicorn
from fastapi import FastAPI, Request
from BankNotes import BankNote
import pickle
import pandas as pd
import numpy as np
import os
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load ML model
pickle_in = open('rf_model.pkl', 'rb')
rff = pickle.load(pickle_in)

# Mount static and template folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="frontend")

# Enable CORS (for frontend integration if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
def render_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/hello")
def index():
    return {'message': 'Hello World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to Fastapi app': f"{name} Reddy"}

@app.post('/predict')
def predict_BankNote(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = rff.predict([[variance, skewness, curtosis, entropy]])

    if prediction[0] > 0.5:
        result = 'Fake Note'
    else:
        result = "Bank Note"
    return {"prediction": result}

# Do not run uvicorn here on Render
# Render uses startCommand: uvicorn app:app --host 0.0.0.0 --port 10000
