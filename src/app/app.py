import gradio as gr
import joblib
import yfinance as yf
from src.features import create_features

STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS"
}

models = {
    name: joblib.load(f"models/{ticker}.pkl")
    for name, ticker in STOCKS.items()
}

def predict(stock_name):
    ticker = STOCKS[stock_name]
    model = models[stock_name]

    df = yf.download(ticker, period="1mo")
    df = create_features(df)

    X = df.drop("Close", axis=1)
    pred = model.predict(X)[-1]

    return f"{stock_name} predicted next close price: ₹{pred:.2f}"

app = gr.Interface(
    fn=predict,
    inputs=gr.Dropdown(choices=list(STOCKS.keys()), label="Select Stock"),
    outputs="text",
    title="Indian Stock Price Prediction (MLOps Demo)"
)

app.launch()
