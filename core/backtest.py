#!/usr/bin/env python3
"""
Backtest the model with 5-day trading cycles
"""

import argparse
import os
import pickle
import torch
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from training_system import TrainingSystem

def is_trading_day(date):
    """Check if a date is a trading day (not weekend)."""
    date_obj = datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
    # Monday = 0, Sunday = 6
    return date_obj.weekday() < 5  # Monday-Friday

def get_next_trading_day(date):
    """Get the next available trading day."""
    date_obj = datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
    
    # Skip to next Monday if weekend
    while date_obj.weekday() >= 5:  # Saturday or Sunday
        date_obj += timedelta(days=1)
    
    return date_obj.strftime("%Y-%m-%d")

def get_price(ticker, date, preloaded_data=None, max_lookback=5):
    """Get stock price for a specific date, with fallback to previous trading days."""
    try:
        # Make sure we're looking at a trading day
        date = get_next_trading_day(date)
        
        if preloaded_data is not None:
            # Check if ticker exists in the multi-index structure
            # Data format is (Ticker, Price) like ('BLK', 'Close')
            if (ticker, 'Close') in preloaded_data.columns:
                # Use preloaded data
                price_data = preloaded_data[(ticker, 'Close')]
                
                # Try the exact date first
                if date in price_data.index:
                    price = price_data.loc[date]
                    if not pd.isna(price):
                        return float(price)
                
                # Try looking back up to max_lookback trading days for holidays
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                for i in range(1, max_lookback + 1):
                    lookback_date = date_obj - timedelta(days=i)
                    # Skip weekends
                    if lookback_date.weekday() >= 5:
                        continue
                    lookback_str = lookback_date.strftime("%Y-%m-%d")
                    if lookback_str in price_data.index:
                        price = price_data.loc[lookback_str]
                        if not pd.isna(price):
                            return float(price)
        
        return None
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description='Backtest model with 5-day trading cycles')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital in dollars')
    parser.add_argument('--top-n', type=int, default=3, help='Number of top stocks to buy')
    parser.add_argument('--hold-days', type=int, default=5, help='Days to hold each position')
    args = parser.parse_args()
    
    print(f"🏁 Starting backtest from {args.start} to {args.end}")
    print(f"💰 Starting capital: ${args.capital:,.2f}")
    print(f"📊 Strategy: Buy top {args.top_n} stocks, hold for {args.hold_days} days\n")
    
    # Create training system and preload data
    training_system = TrainingSystem()
    
    # For predictions, use cached data if available
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'backtest_2025.pkl')
    if os.path.exists(cache_path):
        print("📦 Loading cached backtest data for predictions...")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f"✅ Loaded cached data: {cached['start_date']} to {cached['end_date']}")
        
        # Set the cached data in training system for predictions
        # But we need to get the properly structured raw data
        training_system._cached_data = None  # Don't use the malformed cache
    else:
        print("📊 No cached data found")
    
    # For price lookups, download fresh data with proper structure
    print("� Downloading price data for profit calculation...")
    sp500_tickers = training_system.data_processor._get_sp500_tickers()
    preload_start = (datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    preload_end = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
    
    # Use the training system's data processor to get properly formatted data
    price_data_raw = training_system.data_processor.download_market_data(preload_start, preload_end)
    
    # Set this as the cache for predictions
    training_system._cached_data = price_data_raw
    training_system._cache_start = preload_start
    training_system._cache_end = preload_end
    
    print("✅ All data ready!\n")
    
    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained_stock_trader.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_system.model.load_state_dict(torch.load(model_path, map_location=device))
    training_system.model.to(device)
    print(f"✅ Model loaded on {device}\n")
    
    # Run backtest
    portfolio_value = args.capital
    current_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Make sure we start on a trading day
    if current_date.weekday() >= 5:
        current_date_str = get_next_trading_day(current_date)
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    
    all_trades = []
    cycle_num = 0
    skipped_cycles = 0
    
    while current_date <= end_date:
        cycle_num += 1
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Skip weekends
        if not is_trading_day(date_str):
            current_date += timedelta(days=1)
            continue
        
        print(f"{'='*60}")
        print(f"🔄 Cycle {cycle_num}: {date_str}")
        print(f"{'='*60}")
        
        # Get predictions for this date
        try:
            target_stock, top_choices = training_system.predict_action(date_str)
        except Exception as e:
            print(f"⚠️  Could not get predictions for {date_str}: {e}")
            current_date += timedelta(days=1)
            continue
        
        # Select top N stocks
        top_n_stocks = [stock for stock, conf in top_choices[:args.top_n]]
        
        print(f"📊 Top {args.top_n} predictions:")
        for i, (stock, conf) in enumerate(top_choices[:args.top_n], 1):
            print(f"   {i}. {stock}: {conf:.4f}")
        
        # Buy stocks
        per_stock_investment = portfolio_value / args.top_n
        positions = []
        
        print(f"\n💵 Buying ${per_stock_investment:,.2f} of each stock...")
        for stock in top_n_stocks:
            buy_price = get_price(stock, date_str, price_data_raw)
            if buy_price is None:
                print(f"   ❌ {stock}: Price not available")
                continue
            
            shares = per_stock_investment / buy_price
            positions.append({
                'stock': stock,
                'buy_date': date_str,
                'buy_price': buy_price,
                'shares': shares,
                'investment': per_stock_investment
            })
            print(f"   ✅ {stock}: Buy at ${buy_price:.2f} ({shares:.2f} shares)")
        
        if not positions:
            print("   ⚠️  No valid positions, skipping cycle")
            skipped_cycles += 1
            current_date += timedelta(days=args.hold_days)
            continue
        
        # Calculate actual sell date (skip weekends and add buffer for holidays)
        sell_date = current_date + timedelta(days=args.hold_days)
        sell_date_str = get_next_trading_day(sell_date)
        
        print(f"\n💰 Selling on {sell_date_str} after ~{args.hold_days}-day hold...")
        
        cycle_pnl = 0
        valid_sells = 0
        for pos in positions:
            sell_price = get_price(pos['stock'], sell_date_str, price_data_raw, max_lookback=10)
            
            if sell_price is None:
                print(f"   ⚠️  {pos['stock']}: Could not get sell price (skipping)")
                # Return capital for failed sell
                portfolio_value += pos['investment']
                continue
            
            proceeds = pos['shares'] * sell_price
            pnl = proceeds - pos['investment']
            pnl_pct = (pnl / pos['investment']) * 100
            cycle_pnl += pnl
            valid_sells += 1
            
            trade = {
                'cycle': cycle_num,
                'stock': pos['stock'],
                'buy_date': pos['buy_date'],
                'sell_date': sell_date_str,
                'buy_price': pos['buy_price'],
                'sell_price': sell_price,
                'shares': pos['shares'],
                'investment': pos['investment'],
                'proceeds': proceeds,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            all_trades.append(trade)
            
            emoji = "📈" if pnl >= 0 else "📉"
            print(f"   {emoji} {pos['stock']}: ${pos['buy_price']:.2f} → ${sell_price:.2f} = {pnl_pct:+.2f}% (${pnl:+.2f})")
        
        if valid_sells > 0:
            portfolio_value += cycle_pnl
            print(f"\n💼 Cycle P&L: ${cycle_pnl:+.2f}")
            print(f"💰 Portfolio Value: ${portfolio_value:,.2f}\n")
        else:
            print(f"\n⚠️  No valid sells in this cycle\n")
        
        # Move to next cycle (skip to next trading day)
        current_date = datetime.strptime(sell_date_str, "%Y-%m-%d") + timedelta(days=1)
        current_date_str = get_next_trading_day(current_date)
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST SUMMARY")
    print(f"{'='*60}")
    
    total_pnl = portfolio_value - args.capital
    total_return = (total_pnl / args.capital) * 100
    
    print(f"Starting Capital:    ${args.capital:,.2f}")
    print(f"Ending Portfolio:    ${portfolio_value:,.2f}")
    print(f"Total P&L:           ${total_pnl:+,.2f}")
    print(f"Total Return:        {total_return:+.2f}%")
    print(f"Number of Cycles:    {cycle_num}")
    print(f"Skipped Cycles:      {skipped_cycles}")
    print(f"Total Trades:        {len(all_trades)}")
    
    if all_trades:
        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        losing_trades = [t for t in all_trades if t['pnl'] < 0]
        
        print(f"\nWinning Trades:      {len(winning_trades)} ({len(winning_trades)/len(all_trades)*100:.1f}%)")
        print(f"Losing Trades:       {len(losing_trades)} ({len(losing_trades)/len(all_trades)*100:.1f}%)")
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        print(f"Average Win:         ${avg_win:+,.2f}")
        print(f"Average Loss:        ${avg_loss:+,.2f}")
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main()
