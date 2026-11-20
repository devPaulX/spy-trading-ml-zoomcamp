from pathlib import Path
import pandas as pd
import yfinance as yf

START_DATE = "2015-01-01"
OUTPUT = Path("data/market_data.csv")


def load_data():
    # Download SPY prices
    spy = yf.download(
        "SPY",
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if spy.empty:
        raise RuntimeError("Yahoo Finance returned no SPY data")

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = spy.reset_index().rename(columns={
        "Date": "date",
        "Open": "spy_open",
        "High": "spy_high",
        "Low": "spy_low",
        "Close": "spy_close",
        "Volume": "spy_volume",
    })

    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()

    # Download Treasury yields from FRED
    url = (
        "https://fred.stlouisfed.org/graph/"
        "fredgraph.csv?id=DGS10"
    )

    fred = pd.read_csv(url, na_values=["."])
    fred = fred.iloc[:, :2]
    fred.columns = ["date", "us10y_yield"]
    fred["date"] = pd.to_datetime(fred["date"])
    fred["us10y_yield"] = pd.to_numeric(
        fred["us10y_yield"], errors="coerce"
    )

    # Combine by trading date
    df = spy.merge(fred, on="date", how="left")
    df = df.sort_values("date")
    df["us10y_yield"] = df["us10y_yield"].ffill()
    df = df.dropna(subset=["us10y_yield"])

    return df


if __name__ == "__main__":
    df = load_data()
    assert len(df) > 1000
    assert df["spy_close"].notna().all()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"DATA READY: {len(df)} rows")
    print(f"DATE RANGE: {df.date.min()} to {df.date.max()}")
    print(f"COLUMNS: {list(df.columns)}")
    print(f"SAVED: {OUTPUT}")
