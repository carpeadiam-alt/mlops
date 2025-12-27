import pandas as pd

def create_features(df, lags=5):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df.dropna(inplace=True)
    return df
