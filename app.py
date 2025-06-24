# importing Required Libraries
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import pickle
import pandas as pd
import numpy as np

# starting fastapi app

app = FastAPI()
pickle_in = open('rf_model.pkl', 'rb')
rff = pickle.load(pickle_in)

# index route
@app.get('/')
def index():
    return {'message' : 'Hello World'}

# name route
@app.get('/{name}')
def get_name(name:str):
    return {'Welcome to Fastapi app' : f"{name} Reddy"}

# expose predict functionality
@app.post('/predict')
def predict_BankNote(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = rff.predict([[variance, skewness, curtosis, entropy]])

    if prediction[0]>0.5:
        prediction = 'Fake Note'
    else:
        prediction = "Bank Note"
    return{
        "prediction" : prediction
    }
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
