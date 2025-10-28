#!/usr/bin/env python3
"""
Backtest the model with 5-day trading cycles
"""

import argparse
import os
import torch
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from training_system import TrainingSystem

def get_price(ticker, date, preloaded_data=None):
    """Get stock price for a specific date."""
    try:
        if preloaded_data is not None and ticker in preloaded_data.columns.get_level_values(1):
            # Use preloaded data
            price_data = preloaded_data['Close'][ticker]
            if date in price_data.index:
                price = price_data.loc[date]
                return float(price) if not pd.isna(price) else None
        
        # Fallback to yfinance download
        data = yf.download(ticker, start=date, 
                          end=(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d"), 
                          progress=False)
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if 'Close' not in data.columns:
            return None
            
        price = data['Close'].iloc[0]
        return float(price) if not pd.isna(price) else None
    except:
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
    
    # Preload data (need extra days before start for model input)
    preload_start = (datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    preload_end = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
    
    print("📊 Preloading market data (avoids repeated downloads)...")
    training_system.preload_data(preload_start, preload_end)
    
    # Also preload price data for the backtest period
    print("📈 Preloading price data for profit calculation...")
    sp500_tickers = training_system.data_processor._get_sp500_tickers()
    price_data = yf.download(sp500_tickers, start=args.start, end=preload_end, progress=False)
    print("✅ All data preloaded!\n")
    
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
    
    all_trades = []
    cycle_num = 0
    
    while current_date <= end_date:
        cycle_num += 1
        date_str = current_date.strftime("%Y-%m-%d")
        
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
            buy_price = get_price(stock, date_str, price_data)
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
            current_date += timedelta(days=args.hold_days)
            continue
        
        # Sell after hold period
        sell_date = current_date + timedelta(days=args.hold_days)
        sell_date_str = sell_date.strftime("%Y-%m-%d")
        
        print(f"\n💰 Selling on {sell_date_str} after {args.hold_days}-day hold...")
        
        cycle_pnl = 0
        for pos in positions:
            sell_price = get_price(pos['stock'], sell_date_str, price_data)
            if sell_price is None:
                # Try next available day
                for offset in range(1, 5):
                    alt_date = (sell_date + timedelta(days=offset)).strftime("%Y-%m-%d")
                    sell_price = get_price(pos['stock'], alt_date, price_data)
                    if sell_price is not None:
                        sell_date_str = alt_date
                        break
            
            if sell_price is None:
                print(f"   ❌ {pos['stock']}: Could not get sell price")
                continue
            
            proceeds = pos['shares'] * sell_price
            pnl = proceeds - pos['investment']
            pnl_pct = (pnl / pos['investment']) * 100
            cycle_pnl += pnl
            
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
        
        portfolio_value += cycle_pnl
        print(f"\n💼 Cycle P&L: ${cycle_pnl:+.2f}")
        print(f"💰 Portfolio Value: ${portfolio_value:,.2f}\n")
        
        # Move to next cycle
        current_date = sell_date + timedelta(days=1)
    
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
