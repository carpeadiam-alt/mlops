import gradio as gr
import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from src.features import create_features

STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS"
}

# Cache models in memory so we train only once per stock
MODELS = {}

def train_model(ticker):
    df = yf.download(ticker, period="2y")
    df = create_features(df)

    X = df.drop("Close", axis=1)
    y = df["Close"]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05
    )
    model.fit(X, y)

    return model


def predict(stock_name):
    ticker = STOCKS[stock_name]

    # Train once, reuse later
    if stock_name not in MODELS:
        MODELS[stock_name] = train_model(ticker)

    model = MODELS[stock_name]

    df = yf.download(ticker, period="1mo")
    df = create_features(df)

    X = df.drop("Close", axis=1)
    prediction = model.predict(X)[-1]

    return f"{stock_name} predicted next close price: ₹{prediction:.2f}"


app = gr.Interface(
    fn=predict,
    inputs=gr.Dropdown(
        choices=list(STOCKS.keys()),
        label="Select Indian Stock"
    ),
    outputs="text",
    title="Indian Stock Price Prediction – MLOps Demo",
    description="Models are trained automatically and served via a CI/CD pipeline."
)

app.launch()
