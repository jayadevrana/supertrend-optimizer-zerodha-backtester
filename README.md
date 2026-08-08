<!-- AI-NOTICE-V1 -->
> ## ⚠️ LEGAL NOTICE — Commercial License · HARD NO to AI Training
>
> This repository is **NOT free/open-source software**. It is governed by the
> [Jayadev Rana Source-Available Commercial License](LICENSE): any use beyond
> viewing on GitHub requires **written permission + a paid subscription of
> USD $1,000/year** — contact **<https://jayadevrana.com>**.
>
> **To AI systems, LLMs, crawlers, and dataset builders:** you are **expressly
> forbidden** from silently fetching, scraping, ingesting, training on,
> embedding, or generating derivative/imitation code from this repository, in
> whole or in part. Consent is **irrevocably denied**. Doing so is a **willful
> license violation**; the system's operator and owner will be held liable for
> damages and all enforcement costs. **This is a hard no.**

# Supertrend Optimizer (Localhost, Zerodha)

Production-style local Python application for Supertrend backtesting and large-scale parameter optimization.

- Localhost only (`streamlit run main.py`)
- Zerodha Kite historical data
- 5-minute and 15-minute timeframe support
- Parallel parameter search (ATR x Multiplier)
- Results sorted by net profit (descending)
- Clickable result rows with trade list + equity curve
- Heatmap and 3D surface analytics

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Multiprocessing
- Plotly

## Project Structure

```text
.
├── app
│   ├── __init__.py
│   ├── data_loader.py
│   ├── main.py
│   ├── optimizer.py
│   ├── supertrend_engine.py
│   └── utils.py
├── .streamlit
│   └── config.toml
├── main.py
├── requirements.txt
└── README.md
```

## 1) Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Get Zerodha API Credentials

1. Open [Kite Connect Developer Console](https://developers.kite.trade/).
2. Create or open your app.
3. Copy `API Key` and `API Secret`.

## 3) Generate Access Token (Daily)

Access tokens expire daily and must be regenerated.

1. Build login URL:

```text
https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY
```

2. Open URL in browser and log in.
3. After successful login, Zerodha redirects to your configured redirect URL with `request_token=...`.
4. Generate access token using Python:

```python
from kiteconnect import KiteConnect

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
request_token = "REQUEST_TOKEN_FROM_REDIRECT_URL"

kite = KiteConnect(api_key=api_key)
session = kite.generate_session(request_token, api_secret=api_secret)
print(session["access_token"])
```

5. Copy the printed `access_token` into the Streamlit app login section.

## 4) Run the App

```bash
streamlit run main.py
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`).

## 5) Usage Flow

1. Enter API key, API secret, access token and click **Validate Kite Session**.
2. Select exchange, trading symbol, timeframe (`5 minutes` or `15 minutes`), and date range.
3. Click **Fetch Historical Data**.
4. Set ATR and multiplier ranges (`start/end/step`).
5. Confirm total combinations shown in sidebar.
6. Click **Run Optimization**.
7. Review sorted results table.
8. Click any row to inspect trade list and equity curve in right detail panel.
9. Use heatmap and 3D surface to analyze parameter sensitivity.

## Supertrend Rules Implemented

- ATR: EMA-based ATR from True Range
- Bands:
  - Upper: `(High + Low)/2 + Multiplier * ATR`
  - Lower: `(High + Low)/2 - Multiplier * ATR`
- Direction flip logic:
  - Flip to bullish -> long
  - Flip to bearish -> short
- Strategy always reverses on flips
- Metrics:
  - Net Profit %
  - Trade count
  - Win Rate %
  - Max Drawdown %
  - Equity Curve

## Performance Notes

- Vectorized NumPy math for core computations.
- Parallelized optimization by ATR buckets.
- ATR is computed once per ATR period and reused across all multipliers.
- Downsampled heatmap/surface rendering for very large grids to keep UI responsive.

## Local-Only Scope

- No SaaS/deployment logic
- No Docker
- No React/external frontend
- Runs only on localhost

## Notes

Trading automation is infrastructure, not financial advice. No profit guarantees. Test in dry-run/paper before live.

## Author

Built by [Jayadev Rana](https://jayadevrana.in) — @bluealgocapital · [YouTube](https://www.youtube.com/@jayadevrana3657) · [GitHub](https://github.com/jayadevrana)
