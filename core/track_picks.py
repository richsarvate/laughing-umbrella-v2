#!/usr/bin/env python3
"""
Track model's top pick every 5 days and see how each performs over time
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
    parser = argparse.ArgumentParser(description='Track model picks every 5 days over a month')
    parser.add_argument('--start', type=str, default='2025-01-02', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=int, default=5, help='Days between predictions')
    args = parser.parse_args()
    
    print(f"🏁 Tracking model performance from {args.start} to {args.end}")
    print(f"📊 Making a new prediction every {args.interval} days\n")
    
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
    
    # Track predictions over time
    current_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    picks = []
    pick_num = 0
    
    # Make predictions every N days
    while current_date <= end_date:
        pick_num += 1
        date_str = current_date.strftime("%Y-%m-%d")
        
        print(f"{'='*70}")
        print(f"📅 Pick #{pick_num}: {date_str}")
        print(f"{'='*70}")
        
        # Get prediction
        try:
            target_stock, top_choices = training_system.predict_action(date_str)
        except Exception as e:
            print(f"⚠️  Could not get predictions for {date_str}: {e}\n")
            current_date += timedelta(days=args.interval)
            continue
        
        print(f"🎯 Top prediction: {target_stock} (confidence: {top_choices[0][1]:.4f})")
        print(f"📊 Top 3: {top_choices[0][0]} ({top_choices[0][1]:.4f}), "
              f"{top_choices[1][0]} ({top_choices[1][1]:.4f}), "
              f"{top_choices[2][0]} ({top_choices[2][1]:.4f})")
        
        # Get price at prediction time
        entry_price = get_price(target_stock, date_str, price_data)
        if entry_price is None:
            print(f"❌ Could not get price for {target_stock}\n")
            current_date += timedelta(days=args.interval)
            continue
        
        print(f"💵 Entry price: ${entry_price:.2f}")
        
        # Track performance over time until end date
        print(f"\n📈 Performance tracking:")
        pick_data = {
            'pick_num': pick_num,
            'date': date_str,
            'stock': target_stock,
            'entry_price': entry_price,
            'confidence': top_choices[0][1],
            'performance': []
        }
        
        # Check price every day from entry to end
        check_date = current_date
        while check_date <= end_date:
            check_str = check_date.strftime("%Y-%m-%d")
            current_price = get_price(target_stock, check_str, price_data)
            
            if current_price is not None:
                days_held = (check_date - current_date).days
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                pick_data['performance'].append({
                    'date': check_str,
                    'days': days_held,
                    'price': current_price,
                    'pnl_pct': pnl_pct
                })
                
                if days_held % 5 == 0 and days_held > 0:  # Show every 5 days
                    emoji = "📈" if pnl_pct >= 0 else "📉"
                    print(f"   Day {days_held:2d} ({check_str}): ${current_price:.2f} = {emoji} {pnl_pct:+.2f}%")
            
            check_date += timedelta(days=1)
        
        # Show final performance
        if pick_data['performance']:
            final = pick_data['performance'][-1]
            emoji = "📈" if final['pnl_pct'] >= 0 else "📉"
            print(f"\n🏁 Final ({final['date']}): ${final['price']:.2f} = {emoji} {final['pnl_pct']:+.2f}% "
                  f"(held {final['days']} days)")
        
        picks.append(pick_data)
        print()
        
        # Move to next prediction date
        current_date += timedelta(days=args.interval)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY: {len(picks)} predictions over {(end_date - datetime.strptime(args.start, '%Y-%m-%d')).days} days")
    print(f"{'='*70}\n")
    
    for pick in picks:
        if pick['performance']:
            final_pnl = pick['performance'][-1]['pnl_pct']
            days_held = pick['performance'][-1]['days']
            emoji = "✅" if final_pnl >= 0 else "❌"
            
            print(f"{emoji} Pick #{pick['pick_num']} ({pick['date']}): {pick['stock']} "
                  f"${pick['entry_price']:.2f} → ${pick['performance'][-1]['price']:.2f} = "
                  f"{final_pnl:+.2f}% over {days_held} days")
    
    # Overall stats
    if picks:
        final_returns = [p['performance'][-1]['pnl_pct'] for p in picks if p['performance']]
        if final_returns:
            avg_return = sum(final_returns) / len(final_returns)
            winners = len([r for r in final_returns if r > 0])
            
            print(f"\n📈 Average return: {avg_return:+.2f}%")
            print(f"🎯 Win rate: {winners}/{len(final_returns)} ({winners/len(final_returns)*100:.1f}%)")
            print(f"📊 Best pick: {max(final_returns):+.2f}%")
            print(f"📊 Worst pick: {min(final_returns):+.2f}%")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
