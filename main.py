import os
import numpy as np
import pandas as pd

# Import our modular plug-and-play components
# (Ensure data_loader.py, indicators.py, ranking.py, backtest.py, and storage.py are in the same directory)
from data_loader import R2DataLoader
from indicators import compute_panel_indicators
from ranking import compute_rankings_and_top100  # From Chat 3 output logic
from backtest import BacktestEngine
from storage import R2StorageManager

def run_pipeline():
    # 1. Configuration & Credentials (pulling from environment variables or custom config)
    ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
    ACCESS_KEY = "00e18b0c16ecb3395cd6f7c8e0eb3554"
    SECRET_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
    BUCKET_NAME = "stocks-data"

    
    print("Step 1: Initializing Data Loader and fetching from Cloudflare R2...")
    loader = R2DataLoader(
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        bucket_name=BUCKET_NAME,
        sample_size=2
    )
    
    dataset = loader.process_dataset()
    timestamps = dataset["timestamps"]
    valid_tickers = dataset["valid_tickers"]
    panel = dataset["panel"]  # Shape: (Num_Tickers, Timestamps, 5)
    
    print(f"-> Complete dataset loaded. Valid Tickers: {len(valid_tickers)}, Timestamps: {len(timestamps)}")

    print("\nStep 2: Computing Multi-Period Performance & Indicators (Chat 2 Logic)...")
    indicators = compute_panel_indicators(panel, timestamps)
    avg_ema_matrix = indicators["avg_ema"]  # List 1 metric
    vol_sma_matrix = indicators["vol_sma"]  # List 2 metric

    print("\nStep 3: Calculating Cross-Sectional Rankings & Top 100 Selection (Chat 3 Logic)...")
    # Note: Ensure your ranking.py script exposes a function matching this signature
    ranking_results = compute_rankings_and_top100(avg_ema_matrix, vol_sma_matrix)
    list1_ranks = ranking_results["list1_ranks"]
    list2_ranks = ranking_results["list2_ranks"]
    list3_ranks = ranking_lists = ranking_results["list3_ranks"]
    top_100_matrix = ranking_results["top_100_matrix"] # Boolean mask: (Num_Tickers, Timestamps)

    print("\nStep 4: Executing Portfolio Backtest & 1% Weight Accounting (Chat 4 Logic)...")
    close_prices = panel[:, :, 3]  # Extract close price matrix
    backtester = BacktestEngine(fixed_weight=0.01)
    backtest_results = backtester.run(
        top_100_mask=top_100_matrix,
        close_prices=close_prices,
        timestamps=timestamps,
        tickers=valid_tickers
    )
    
    equity_df = backtest_results["equity_df"]
    trade_log_df = backtest_results["trade_log"]
    print(f"-> Backtest complete. Total trades logged: {len(trade_log_df)}")

    print("\nStep 5: Serializing and Uploading Results back to R2 stats/ folder (Chat 5 Logic)...")
    storage = R2StorageManager(
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        bucket_name=BUCKET_NAME,
        stats_prefix="stats/"
    )
    
    # Upload valid tickers JSON
    storage.save_valid_tickers(valid_tickers)
    
    # Upload daily top 100 ordered stock list CSV
    storage.save_top_100_rankings(timestamps, valid_tickers, top_100_matrix)
    
    # Upload individual list ranks CSV
    storage.save_individual_ranks(timestamps, valid_tickers, list1_ranks, list2_ranks, list3_ranks)
    
    # Upload daily portfolio contribution equity curve CSV
    storage.save_portfolio_equity_curve(
        timestamps=timestamps,
        equity_curve=backtest_results["equity_curve_array"],
        daily_contributions=backtest_results["daily_returns_array"]
    )
    
    print("\nPipeline execution successfully finished and uploaded to Cloudflare R2!")

if __name__ == "__main__":
    run_pipeline()