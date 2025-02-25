# app/main.py
from fastapi import FastAPI
from .predict import predict_animal

app = FastAPI()

app.include_router(predict_animal)
