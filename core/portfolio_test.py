#!/usr/bin/env python3
"""
5-day cycle portfolio test
"""

import random
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from training_system import TrainingSystem

def get_price(ticker, date):
    """Get stock price for a date."""
    try:
        data = yf.download(ticker, start=date, 
                          end=(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d"), 
                          progress=False)
        if data.empty:
            return None
        
        # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single tickers)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if 'Close' not in data.columns:
            return None
            
        price = data['Close'].iloc[0]
        return float(price) if not pd.isna(price) else None
    except:
        return None

def main():
    print("🎯 5-DAY CYCLE PORTFOLIO TEST")
    print("=" * 50)
    
    # Random month in 2025
    month = random.randint(1, 9)
    start_date = datetime(2025, month, 1)
    
    # Generate 4 cycles of 5 trading days each (approximately 1 month)
    cycles = []
    current = start_date
    
    for cycle_num in range(4):
        cycle_days = []
        days_added = 0
        
        while days_added < 5:
            if current.weekday() < 5:  # Weekday only
                cycle_days.append(current.strftime("%Y-%m-%d"))
                days_added += 1
            current += timedelta(days=1)
        
        cycles.append(cycle_days)
    
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    print(f"Testing: {month_names[month]} 2025")
    print(f"Cycles: {len(cycles)}")
    print(f"Starting: $10,000")
    print("=" * 50)
    
    # Portfolio state
    trader = TrainingSystem()
    portfolio = 10000.0
    position = "CASH"
    shares = 0.0
    
    # Preload all data once for efficiency (avoid downloading 4 times)
    preload_start = (datetime.strptime(cycles[0][0], "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    preload_end = cycles[-1][-1]
    print(f"\nPreloading market data from {preload_start} to {preload_end}...")
    trader.preload_data(preload_start, preload_end)
    
    # Run each 5-day cycle
    for i, cycle in enumerate(cycles, 1):
        decision_date = cycle[0]
        end_date = cycle[-1]
        
        print(f"\nCycle {i}: {decision_date} to {end_date}")
        
        # Get model decision (returns selected stock + top 3 choices)
        try:
            stock, top3_choices = trader.predict_action(decision_date)
            print(f"  Selected: {stock}")
        except Exception as e:
            print(f"  Error: {str(e)}")
            stock = None
            top3_choices = []
        
        # Execute trade: sell current and buy selected stock
        if stock and position != stock:
            # Sell current position
            if position != "CASH":
                sell_price = get_price(position, decision_date)
                if sell_price:
                    portfolio = shares * sell_price
                    print(f"  Sold {shares:.0f} shares of {position} @ ${sell_price:.2f}")
            
            # Try top 3 predictions in order until one succeeds
            trade_executed = False
            for choice_stock, choice_prob in top3_choices:
                buy_price = get_price(choice_stock, decision_date)
                if buy_price:
                    shares = portfolio / buy_price
                    position = choice_stock
                    print(f"  Bought {shares:.0f} shares of {choice_stock} @ ${buy_price:.2f} (confidence: {choice_prob:.3f})")
                    trade_executed = True
                    break
                else:
                    print(f"  Failed to buy {choice_stock} (delisted/unavailable), trying next...")
            
            if not trade_executed:
                position = "CASH"
                shares = 0
                print(f"  All stock picks failed, holding cash")
        
        elif stock == position:
            # HOLD: already in the selected stock
            print(f"  Holding {position} (no change)")
        
        # Evaluate at end of cycle
        if position != "CASH":
            end_price = get_price(position, end_date)
            if end_price:
                end_value = shares * end_price
                cycle_return = (end_value / portfolio - 1) * 100
                print(f"  End: ${end_value:.2f} ({cycle_return:+.1f}%)")
                portfolio = end_value
            else:
                print(f"  End: Price error for {position}")
        else:
            print(f"  End: ${portfolio:.2f} (cash)")
    
    # Final results
    total_return = (portfolio / 10000 - 1) * 100
    
    print("=" * 50)
    print("RESULTS")
    print(f"Start:  $10,000")
    print(f"End:    ${portfolio:.2f}")
    print(f"Return: {total_return:+.1f}%")
    print(f"Final:  {position}")
    print("=" * 50)

if __name__ == "__main__":
    main()