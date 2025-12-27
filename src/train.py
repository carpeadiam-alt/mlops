import yfinance as yf
import mlflow
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from features import create_features

STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS"
}

os.makedirs("models", exist_ok=True)
mlflow.set_experiment("indian-stock-prediction")

def train_stock(name, ticker):
    df = yf.download(ticker, period="2y")
    df = create_features(df)

    X = df.drop("Close", axis=1)
    y = df["Close"]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05
    )

    with mlflow.start_run(run_name=name):
        model.fit(X, y)
        preds = model.predict(X)

        rmse = mean_squared_error(y, preds, squared=False)

        mlflow.log_param("stock", name)
        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("rmse", rmse)

        path = f"models/{ticker}.pkl"
        joblib.dump(model, path)
        mlflow.log_artifact(path)

if __name__ == "__main__":
    for name, ticker in STOCKS.items():
        train_stock(name, ticker)
