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
        df.columns = ["Close","High","Low","Open","Adj Close","Volume"]
    elif df.shape[1] == 5:
        df.columns = ["Close","High","Low","Open","Volume"]
    else:
        raise RuntimeError(f"Unexpected columns: {list(df.columns)}")

    # 3) DatetimeIndex (tz-naive), slice to END_DATE, keep last 120 bars
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df.index <= pd.to_datetime(end)].tail(120).copy()

    # 4) EMAs
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    st.dataframe(df.tail(), use_container_width=True)
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
    "Based on the image provided identify the trading chart patterns and provide the dates and give reasoning\n"
    "The charts should be mainly among these patterns:\n\n"
    "Step Chart – Shows price moving in distinct horizontal and vertical steps, reflecting consolidation phases followed by sharp directional moves.\n\n"
    "Flag – A short consolidation channel after a strong trend, signaling continuation once the flag breaks in the trend’s direction.\n\n"
    "Pennant – A small symmetrical triangle following a sharp price move, indicating trend continuation after a brief pause.\n\n"
    "1-2-3 Pattern – 1 = sharp up move, 2 = slight move down against the prevailing trend, 3 = break the high of 1 sharp move.\n\n"
    "Cup and Handle – A rounded U shape followed by a small dip, suggesting accumulation before a bullish breakout.\n\n"
    "Triangle – Converging trendlines showing price compression, typically resolving in a breakout aligned with the prevailing trend.\n\n"
    "Low Cheat – After an uptrend a slight correction and a small range consolidation. Once the price breaks out of this consolidation its a buy signal/\n\n"
    "VCP (Volatility Contraction Pattern) – Series of smaller pullbacks with decreasing volatility and volume, showing supply drying up before a high-volume breakout from resistance.\n\n"
    "Inverted Head and Shoulders – A three-trough reversal formation where the middle trough is deepest, signaling a bullish reversal.\n\n"
    "Engulfing / Reversal – A candlestick where a larger candle fully covers the previous one,with close of the current candle below the last candle low  possibly signaling a trend change, especially at key support/resistance.\n\n"
    "Leave blank or mention if no pattern is found if none of these patterns are found in the chart.\n"
)
# PATTERN_PROMPT = (
#     "You are provided the actual chart image and image of different patterns. Based on it identify the different patterns that are forming along with the dates when they are formeed. Do provide reasoning for each of the chart and patterns\n"
#      "Leave blank or mention if no pattern is found if none of these patterns are found in the chart.\n"
# )
# PATTERN_PROMPT=f"""
# You are provided the actual chart image Based on it identify the different patterns that are forming along with the dates when they are formeed. Do provide reasoning for each of the chart and patterns\
# Flag Pattern:
# - Identify a strong trend: Look for a sharp price move in one direction forming the flagpole.
# - Observe consolidation: Price then moves sideways in a narrow rectangular or slight channel, forming the flag. The consolidation is brief (few days to weeks).
# - Volume behavior: High volume during the flagpole, reduced volume during consolidation, and increased volume on breakout.
# - Types:
#     • Bullish Flag – Upward flagpole, slight downward/sideways consolidation. Continuation expected upward.
#     • Bearish Flag – Downward flagpole, slight upward/sideways consolidation. Continuation expected downward.
# - Trading Use:
#     • Entry – Long above flag resistance or short below flag support.
#     • Target – Measure flagpole length and project from breakout.
#     • Stop-loss – Below flag’s lower trendline for bullish, above upper trendline for bearish.

# Pennant Pattern:
# - Structure: Strong, fast price move first creates the flagpole, followed by a small symmetrical triangle (pennant) formed by converging trendlines.
# - Breakout: Typically occurs in the same direction as the prior trend, confirming continuation.
# - Key Characteristics:
#     • Shape – A tiny symmetrical triangle (unlike the parallel lines of a flag).
#     • Volume – Heavy during flagpole, low inside pennant, and surges at breakout.
#     • Duration – Ideally 1–4 weeks; beyond 12 weeks becomes a symmetrical triangle.
# - Purpose: Used to anticipate continuation of a prevailing trend and identify clean entry/exit points.
# Triangle Pattern
# How to Trade Triangle Patterns
# Traders use triangle patterns to identify potential entry and exit points and manage risk. The general approach involves waiting for a confirmed breakout (or breakdown) before entering a trade. 
# Confirmation A breakout is considered more reliable when it is accompanied by an increase in trading volume and a decisive close beyond the trendline.
# Entry Point For an ascending triangle, a trader would typically enter a long position after the price breaks above the resistance line. For a descending triangle, a short position would be considered after the price breaks below the support line. For a symmetrical triangle, the entry is in the direction of the confirmed breakout.
# Stop-Loss and Price Target A stop-loss order is usually placed just outside the opposite side of the breakout to manage risk. The potential price target after a breakout is often estimated by measuring the height of the widest part of the triangle and projecting that distance from the breakout point.

# Cup and handle: 

# A fall from swing high Price level A. Gradually strength of toward down side get reduces. And gradually stock start recovering. Recovery must be gradual U shaped recovery from bottom which I will quote as price level B. recovery will continue till the point Price level A an made almost same high . Recover with V shape must not be consider.

# after recovery stock must recover up to previous swing high  price level A . And again we see small deep on chart . deep on the must be less than 50% of, recovery from the bottom price level B to high price level A. 

# handle will complete on the on price point A. and form one flat line jointing all price point A.



# Volume must contract during all downside movement in cup and handle. And should expand while price recovery is happening in the form of U Shap.
# 123 pattern
# How the pattern works
# Point 1: The initial highest point in a downtrend or the lowest point in an uptrend.
# Point 2: The first retracement from point 1, forming a lower high in a downtrend or a higher low in an uptrend.
# Point 3: The point where the price pulls back again but does not go as far as point 1. The pattern is confirmed, and a trade entry is signaled, once the price moves past point 2.

# Step chart

# Ascending step chart pattern (Uptrend)
# Formation: The price makes a series of higher highs and higher lows, creating an upward "staircase" effect.
# Market psychology: This pattern suggests that buying pressure is dominant, leading to a sustained upward trend.
# Example: A pattern where a stock price moves up, pulls back, and then moves up again to make a higher high than before, and then pulls back to a higher low. 
# Descending step chart pattern (Downtrend) 
# Formation: The price makes a series of lower highs and lower lows, resembling a downward "staircase".
# Market psychology: This pattern indicates that selling pressure is stronger than buying pressure, and the price is in a downtrend.
# Example: A pattern where a stock price moves down, rallies slightly, and then moves down again to make a lower low than before, and then rallies to a lower high. 
# Key characteristics
# Trend identification: Step patterns are fundamentally a way to visualize the current trend.
# Bullish or bearish: An ascending step pattern is a bullish signal, while a descending one is bearish.
# Strength: The clarity of the steps can indicate the strength of the trend. A clear, consistent pattern shows a strong trend, whereas a messy or unclear pattern may signal a weaker trend or consolidation.


# """



def analyze_with_openai(image_png_bytes: bytes, api_key: str) -> str:
    if not OpenAI:
        raise RuntimeError("openai python package not available.")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    b64 = base64.b64encode(image_png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    # img_path = Path("chart-patterns.jpeg")  # change this
    # with img_path.open("rb") as f:
    #     img_bytes = f.read()
    with open("chart-patterns.jpeg", "rb") as f:
        img_bytes = f.read()

    b64_2 = base64.b64encode(img_bytes).decode("utf-8")
    data_url_2 = f"data:image/png;base64,{b64_2}"

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PATTERN_PROMPT},
                    {"type": "input_image", "image_url": data_url}
                   # {"type": "input_image", "image_url": data_url_2},
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

