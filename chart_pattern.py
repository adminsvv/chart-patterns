import os
import io
import base64
import datetime as dt

import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.lines import Line2D


# OpenAI Responses API
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

def require_login() -> bool:
    """
    Simple username/password gate using Streamlit secrets.
    - Configure in .streamlit/secrets.toml or st.secrets:
        APP_USERNAME = "your_user"
        APP_PASSWORD = "your_password"
    - Returns True if authenticated; otherwise renders a login form and returns False.
    """
    if st.session_state.get("authenticated", False):
        # Show a small logout option
        with st.sidebar:
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()
        return True

    username_secret = st.secrets.get("APP_USERNAME", "admin")
    password_secret = st.secrets.get("APP_PASSWORD", "")

    st.title("Login")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", value="", autocomplete="username")
        p = st.text_input("Password", value="", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if not password_secret:
            st.error("Server is missing APP_PASSWORD in secrets.")
            return False
        if u == username_secret and p == password_secret:
            st.session_state.authenticated = True
            st.success("Logged in successfully.")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


def get_api_key() -> str:
    """Resolve the OpenAI API key from env or Streamlit secrets."""
     # loads from .env if present in this folder
    key = (
         st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    )
    return key or ""


def download_prices(ticker: str, end_date: dt.date, lookback_days: int = 200) -> pd.DataFrame:
    """Download price data and return last ~120 trading sessions up to end_date.

    - Uses yfinance with an exclusive `end` (end_date + 1 day)
    - Ensures columns: Open, High, Low, Close, Volume
    - Adds EMA20 and EMA50
    """
    start = (pd.to_datetime(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.shape[1] == 6:
        df.columns = ["Open","High","Low","Close","Adj Close","Volume"]
    elif df.shape[1] == 5:
        df.columns = ["Open","High","Low","Close","Volume"]
    else:
        raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

    # 3) DatetimeIndex (tz-naive), slice to END_DATE, keep last 120 bars
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df.index <= pd.to_datetime(end)].tail(120).copy()

    # 4) EMAs
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    st.dataframe(df.head(), use_container_width=True)
    return df


def plot_chart(df: pd.DataFrame, ticker: str, end_date: dt.date):
    """Create an mplfinance candlestick chart and return (fig, axes)."""
    aps = []
    if "EMA20" in df.columns:
        aps.append(mpf.make_addplot(df["EMA20"], color="blue", width=1.0))
    if "EMA50" in df.columns:
        aps.append(mpf.make_addplot(df["EMA50"], color="orange", width=1.0))

    fig, axes = mpf.plot(
        df[[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]],
        type="candle",
        style="yahoo",
        volume=True,
        addplot=aps if aps else None,
        datetime_format="%Y-%m-%d",
        xrotation=45,
        title=f"{ticker} — Last 120 Trading Days (ending {end_date})",
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True,
    )

    # Legend (proxy handles) if EMAs plotted
    if aps:
        handles = []
        labels = []
        if "EMA20" in df.columns:
            handles.append(Line2D([0], [0], color="blue", lw=2))
            labels.append("EMA 20")
        if "EMA50" in df.columns:
            handles.append(Line2D([0], [0], color="orange", lw=2))
            labels.append("EMA 50")
        if handles:
            axes[0].legend(handles, labels, loc="upper left")

    return fig, axes


PATTERN_PROMPT = (
    "Based on the image provided identify the trading chart pattern and give reasoning\n"
    "The charts should be mainly among these patterns:\n\n"
    "Step Chart – Shows price moving in distinct horizontal and vertical steps, reflecting consolidation phases followed by sharp directional moves.\n\n"
    "Flag – A short consolidation channel after a strong trend, signaling continuation once the flag breaks in the trend’s direction.\n\n"
    "Pennant – A small symmetrical triangle following a sharp price move, indicating trend continuation after a brief pause.\n\n"
    "1-2-3 Pattern – 1 = sharp up move, 2 = slight move down against the prevailing trend, 3 = break the high of 1 sharp move.\n\n"
    "Cup and Handle – A rounded U shape followed by a small dip, suggesting accumulation before a bullish breakout.\n\n"
    "Triangle – Converging trendlines showing price compression, typically resolving in a breakout aligned with the prevailing trend.\n\n"
    "Low Cheat – An early breakout entry setup within a base, entered near support before full pattern confirmation.\n\n"
    "VCP (Volatility Contraction Pattern) – Series of smaller pullbacks with decreasing volatility and volume, showing supply drying up before a high-volume breakout from resistance.\n\n"
    "Inverted Head and Shoulders – A three-trough reversal formation where the middle trough is deepest, signaling a bullish reversal.\n\n"
    "Engulfing / Reversal – A candlestick where a larger candle fully covers the previous one, possibly signaling a trend change, especially at key support/resistance.\n\n"
    "Leave blank or mention if no pattern is found if none of these patterns are found in the chart.\n"
)


def analyze_with_openai(image_png_bytes: bytes, api_key: str) -> str:
    if not OpenAI:
        raise RuntimeError("openai python package not available.")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    b64 = base64.b64encode(image_png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PATTERN_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    # New Responses API exposes convenience property
    output_text = getattr(response, "output_text", str(response))
    usage = getattr(response, "usage", None)
    return output_text, usage


def main():
    st.set_page_config(page_title="Chart Pattern Analyzer", layout="wide")
    st.title("Chart Pattern Analyzer (Streamlit)")
    if not require_login():
        st.stop()

    # Inputs
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        raw = st.text_input("Stock symbol (e.g., RELIANCE)", value="RELIANCE")
    with col2:
        end_date = st.date_input("End date", value=dt.date.today())
    with col3:
        run_btn = st.button("Analyze")

    if not raw:
        st.info("Enter a stock symbol to begin.")
        return

    # Build NSE ticker (avoid double suffix)
    symbol = raw.strip().upper()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    if run_btn:
        try:
            with st.spinner("Downloading data and building chart..."):
                df = download_prices(ticker, end_date)
                fig, _ = plot_chart(df, ticker, end_date)

                # Save to PNG in-memory for both display and OpenAI
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                png_bytes = buf.getvalue()
                buf.close()
            st.image(png_bytes, caption=f"{ticker} up to {end_date}", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to prepare chart: {e}")
            return
        finally:
            try:
                import matplotlib.pyplot as plt  # lazy import to close
                plt.close(fig)
            except Exception:
                pass

        # OpenAI analysis
        api_key = get_api_key()
        if not api_key:
            st.warning("OPENAI_API_KEY not found in environment or Streamlit secrets.")
            return

        try:
            with st.spinner("Asking OpenAI to identify the pattern..."):
                output_text, usage = analyze_with_openai(png_bytes, api_key)
            st.subheader("OpenAI Analysis")
            st.markdown(output_text)

            # Show usage if available
            if usage is not None:
                # Attempt to extract common fields; fallback to printing the object
                #total = getattr(usage, "total_tokens", None)
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)
                st.write("**OpenAI input token cost:**",in_tok*1.6*90/1000000)
                st.write("**OpenAI output token cost:**",out_tok*8*90/1000000)
                
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")


if __name__ == "__main__":

    main()

