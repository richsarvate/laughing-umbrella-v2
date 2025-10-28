#!/usr/bin/env python3
"""
Rolling rebalance: Make a new decision every 5 days, keep or switch stocks
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
            price_data = preloaded_data['Close'][ticker]
            if date in price_data.index:
                price = price_data.loc[date]
                return float(price) if not pd.isna(price) else None
        
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
    parser = argparse.ArgumentParser(description='Rolling rebalance: switch or keep every 5 days')
    parser.add_argument('--start', type=str, default='2025-01-02', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital in dollars')
    parser.add_argument('--rebalance-days', type=int, default=5, help='Days between rebalance decisions')
    args = parser.parse_args()
    
    print(f"🏁 Rolling rebalance strategy from {args.start} to {args.end}")
    print(f"💰 Starting capital: ${args.capital:,.2f}")
    print(f"🔄 Rebalance every {args.rebalance_days} days (switch or keep)\n")
    
    # Create training system and preload data
    training_system = TrainingSystem()
    
    preload_start = (datetime.strptime(args.start, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    preload_end = (datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
    
    print("📊 Preloading market data...")
    training_system.preload_data(preload_start, preload_end)
    
    print("📈 Preloading price data...")
    sp500_tickers = training_system.data_processor._get_sp500_tickers()
    price_data = yf.download(sp500_tickers, start=args.start, end=preload_end, progress=False)
    print("✅ All data preloaded!\n")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained_stock_trader.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_system.model.load_state_dict(torch.load(model_path, map_location=device))
    training_system.model.to(device)
    print(f"✅ Model loaded on {device}\n")
    
    # Track portfolio
    portfolio_value = args.capital
    current_stock = None
    current_shares = 0
    current_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    decision_num = 0
    all_decisions = []
    
    print(f"{'='*70}")
    
    while current_date <= end_date:
        decision_num += 1
        date_str = current_date.strftime("%Y-%m-%d")
        
        print(f"\n📅 Decision #{decision_num}: {date_str}")
        print(f"💰 Portfolio value: ${portfolio_value:,.2f}")
        
        # Get model's top prediction
        try:
            target_stock, top_choices = training_system.predict_action(date_str)
        except Exception as e:
            print(f"⚠️  Could not get predictions: {e}")
            current_date += timedelta(days=args.rebalance_days)
            continue
        
        print(f"🎯 Model recommends: {target_stock} (confidence: {top_choices[0][1]:.4f})")
        
        # If we already hold this stock, keep it
        if current_stock == target_stock:
            print(f"✅ KEEP {current_stock} (already holding)")
            
        # If we hold a different stock or nothing, switch
        else:
            # Sell current position if we have one
            if current_stock is not None:
                sell_price = get_price(current_stock, date_str, price_data)
                if sell_price is None:
                    print(f"❌ Could not get price for {current_stock}, keeping position")
                    current_date += timedelta(days=args.rebalance_days)
                    continue
                
                proceeds = current_shares * sell_price
                pnl = proceeds - portfolio_value
                pnl_pct = (pnl / portfolio_value) * 100
                
                emoji = "📈" if pnl >= 0 else "📉"
                print(f"{emoji} SELL {current_stock}: ${portfolio_value/current_shares:.2f} → ${sell_price:.2f} "
                      f"({current_shares:.2f} shares) = {pnl_pct:+.2f}% (${pnl:+,.2f})")
                
                portfolio_value = proceeds
            
            # Buy new stock
            buy_price = get_price(target_stock, date_str, price_data)
            if buy_price is None:
                print(f"❌ Could not get price for {target_stock}")
                current_stock = None
                current_shares = 0
                current_date += timedelta(days=args.rebalance_days)
                continue
            
            current_shares = portfolio_value / buy_price
            current_stock = target_stock
            
            print(f"🔄 BUY {target_stock}: ${buy_price:.2f} ({current_shares:.2f} shares)")
            
            all_decisions.append({
                'decision': decision_num,
                'date': date_str,
                'action': 'SWITCH' if decision_num > 1 else 'INITIAL',
                'stock': target_stock,
                'price': buy_price,
                'portfolio_value': portfolio_value
            })
        
        # Move to next decision date
        current_date += timedelta(days=args.rebalance_days)
    
    # Final liquidation
    print(f"\n{'='*70}")
    print(f"🏁 FINAL LIQUIDATION: {end_date.strftime('%Y-%m-%d')}")
    
    if current_stock is not None:
        final_date = end_date.strftime("%Y-%m-%d")
        final_price = get_price(current_stock, final_date, price_data)
        
        if final_price is not None:
            final_value = current_shares * final_price
            total_pnl = final_value - args.capital
            total_return = (total_pnl / args.capital) * 100
            
            print(f"💼 Holding: {current_stock} ({current_shares:.2f} shares)")
            print(f"💵 Final price: ${final_price:.2f}")
            print(f"💰 Final portfolio value: ${final_value:,.2f}")
            print(f"\n{'='*70}")
            print(f"📊 PERFORMANCE SUMMARY")
            print(f"{'='*70}")
            print(f"Starting capital:    ${args.capital:,.2f}")
            print(f"Ending value:        ${final_value:,.2f}")
            print(f"Total P&L:           ${total_pnl:+,.2f}")
            print(f"Total return:        {total_return:+.2f}%")
            print(f"Decisions made:      {decision_num}")
            print(f"Stock switches:      {len(all_decisions)}")
            
            # Show decision history
            print(f"\n📋 Decision history:")
            for dec in all_decisions:
                print(f"   {dec['date']}: {dec['action']:8s} → {dec['stock']} @ ${dec['price']:.2f}")
            
            if current_stock:
                print(f"   {final_date}: HOLD     → {current_stock} @ ${final_price:.2f}")
        else:
            print("❌ Could not get final price")
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main()
