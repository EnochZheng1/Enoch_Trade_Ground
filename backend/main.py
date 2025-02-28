from fastapi import FastAPI
import uuid
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()
traders = {}

class Trader:
    def __init__(self, name, strategy="LSTM", balance=10000):
        self.id = str(uuid.uuid4())
        self.name = name
        self.strategy = strategy
        self.balance = balance
        self.portfolio = {}
        self.model = None
        self.scaler = MinMaxScaler()

    def train(self, df):
        self.scaler.fit(df[["Close"]])
        df_scaled = self.scaler.transform(df[["Close"]])
        
        X, y = [], []
        for i in range(60, len(df_scaled)):
            X.append(df_scaled[i-60:i])
            y.append(df_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        
        self.model = model
        return "Model Trained"

    def predict(self, df):
        X_last = df[["Close"]].tail(60).values
        X_last = X_last.reshape(1, 60, 1)
        predicted_scaled = self.model.predict(X_last)[0][0]
        predicted_price = self.scaler.inverse_transform([[predicted_scaled]])[0][0]  # Convert back to original scale
        return float(predicted_price)

@app.post("/create_trader/")
def create_trader(name: str, strategy: str = "LSTM"):
    trader = Trader(name, strategy)
    traders[trader.id] = trader
    return {"trader_id": trader.id, "name": trader.name, "strategy": trader.strategy}

@app.post("/train_trader/")
def train_trader(trader_id: str, stock: str):
    if trader_id not in traders:
        return {"error": "Trader not found"}
    
    df = yf.Ticker(stock).history(period="1y", interval="1d")
    return traders[trader_id].train(df)

@app.post("/predict_trade/")
def predict_trade(trader_id: str, stock: str):
    if trader_id not in traders:
        return {"error": "Trader not found"}
    
    df = yf.Ticker(stock).history(period="1y", interval="1d")
    predicted_price = traders[trader_id].predict(df)
    return {"predicted_price": float(predicted_price)}
