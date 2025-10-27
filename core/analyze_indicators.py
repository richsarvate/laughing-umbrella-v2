#!/usr/bin/env python3
"""
Analyze why the model picks ENPH/NI - check their actual technical indicators
vs other S&P 500 stocks in September 2025.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_system import TrainingSystem

def analyze_stock_indicators():
    """Compare technical indicators of ENPH/NI vs other stocks."""
    print("=" * 80)
    print("Analyzing Why Model Picks ENPH/NI - Technical Indicator Comparison")
    print("=" * 80)
    
    trading_system = TrainingSystem()
    
    # Load data for September 2025
    test_date = "2025-09-15"
    from datetime import datetime, timedelta
    start_date = (datetime.strptime(test_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    
    print(f"\nLoading market data from {start_date} to {test_date}...")
    raw_market_data = trading_system.data_processor.download_market_data(start_date, test_date)
    
    print("Extracting technical features...")
    market_features = trading_system.data_processor.extract_anonymous_features(raw_market_data)
    
    # Get features for the prediction date (last day)
    last_day_features = market_features[-1]  # [num_stocks, 3]
    
    # Create dataframe with all stocks and their features
    stock_data = []
    for i, ticker in enumerate(trading_system.data_processor.sp500_tickers):
        momentum = last_day_features[i, 0]
        volatility = last_day_features[i, 1]
        rsi = last_day_features[i, 2]
        
        stock_data.append({
            'ticker': ticker,
            'momentum_5d': momentum,
            'volatility_20d': volatility,
            'rsi_14d': rsi,
            'combined_score': momentum + volatility + rsi  # Simple combined metric
        })
    
    df = pd.DataFrame(stock_data)
    
    # Sort by different metrics
    print("\n" + "=" * 80)
    print("TOP 10 STOCKS BY MOMENTUM (5-day return):")
    print("=" * 80)
    top_momentum = df.nlargest(10, 'momentum_5d')
    for idx, row in top_momentum.iterrows():
        star = "⭐" if row['ticker'] in ['ENPH', 'NI'] else "  "
        print(f"{star} {row['ticker']:6s} | Momentum: {row['momentum_5d']:+7.3f} | Volatility: {row['volatility_20d']:7.3f} | RSI: {row['rsi_14d']:6.3f}")
    
    print("\n" + "=" * 80)
    print("TOP 10 STOCKS BY VOLATILITY (20-day):")
    print("=" * 80)
    top_volatility = df.nlargest(10, 'volatility_20d')
    for idx, row in top_volatility.iterrows():
        star = "⭐" if row['ticker'] in ['ENPH', 'NI'] else "  "
        print(f"{star} {row['ticker']:6s} | Momentum: {row['momentum_5d']:+7.3f} | Volatility: {row['volatility_20d']:7.3f} | RSI: {row['rsi_14d']:6.3f}")
    
    print("\n" + "=" * 80)
    print("TOP 10 STOCKS BY RSI (14-day):")
    print("=" * 80)
    top_rsi = df.nlargest(10, 'rsi_14d')
    for idx, row in top_rsi.iterrows():
        star = "⭐" if row['ticker'] in ['ENPH', 'NI'] else "  "
        print(f"{star} {row['ticker']:6s} | Momentum: {row['momentum_5d']:+7.3f} | Volatility: {row['volatility_20d']:7.3f} | RSI: {row['rsi_14d']:6.3f}")
    
    # Where do ENPH and NI rank?
    print("\n" + "=" * 80)
    print("ENPH AND NI RANKINGS:")
    print("=" * 80)
    
    for ticker in ['ENPH', 'NI']:
        stock_row = df[df['ticker'] == ticker].iloc[0]
        momentum_rank = (df['momentum_5d'] > stock_row['momentum_5d']).sum() + 1
        volatility_rank = (df['volatility_20d'] > stock_row['volatility_20d']).sum() + 1
        rsi_rank = (df['rsi_14d'] > stock_row['rsi_14d']).sum() + 1
        
        print(f"\n{ticker}:")
        print(f"  Momentum: {stock_row['momentum_5d']:+7.3f} (Rank #{momentum_rank} out of {len(df)})")
        print(f"  Volatility: {stock_row['volatility_20d']:7.3f} (Rank #{volatility_rank} out of {len(df)})")
        print(f"  RSI: {stock_row['rsi_14d']:7.3f} (Rank #{rsi_rank} out of {len(df)})")
    
    # Analysis
    print("\n" + "=" * 80)
    print("💡 ANALYSIS:")
    print("=" * 80)
    
    enph_row = df[df['ticker'] == 'ENPH'].iloc[0]
    ni_row = df[df['ticker'] == 'NI'].iloc[0]
    
    enph_momentum_rank = (df['momentum_5d'] > enph_row['momentum_5d']).sum() + 1
    ni_momentum_rank = (df['momentum_5d'] > ni_row['momentum_5d']).sum() + 1
    
    if enph_momentum_rank <= 10 or ni_momentum_rank <= 10:
        print("✅ ENPH/NI are in TOP 10 for momentum - model picks make sense!")
    elif enph_momentum_rank <= 50 or ni_momentum_rank <= 50:
        print("⚠️  ENPH/NI are in top 50 for momentum - reasonable picks")
    else:
        print("🚨 ENPH/NI are NOT in top rankings - model may have learned wrong patterns!")
        print("   This could indicate:")
        print("   1. Model overfitted to these stocks during training")
        print("   2. Training data had unusual patterns for ENPH/NI that don't generalize")
        print("   3. Model needs retraining on more diverse market conditions")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_stock_indicators()
