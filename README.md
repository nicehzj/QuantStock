# QuantStock

A high-performance, modular quantitative trading analysis workflow system.

## Features
- **Data Source**: Baostock (A-share daily and index data)
- **Storage**: Parquet Data Lake, QuestDB (Time-series), Redis (Cache/Signals)
- **Computation**: DuckDB (OLAP Engine) for fast factor calculation
- **Evaluation**: Alphalens for IC/IR analysis
- **Backtesting**: Vectorbt (Fast screening) & Backtrader (Detailed event-driven)

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python run_pipeline.py`

Refer to `GEMINI.md` for detailed architecture and development conventions.
