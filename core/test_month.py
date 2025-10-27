#!/usr/bin/env python3
"""
Test MC Dropout + Temperature T=5.0 on a random month in 2025.
Downloads data once and runs predictions for all trading days.
"""

import sys
import os
import random
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_system import TrainingSystem

def get_random_month_dates():
    """Get 20 trading days from a random month in 2025."""
    # Random month between Jan-Sep 2025
    month = random.randint(1, 9)
    year = 2025
    
    # Start from first day of month
    start = datetime(year, month, 1)
    
    # Get 20 consecutive weekdays
    trading_days = []
    current = start
    while len(trading_days) < 20:
        if current.weekday() < 5:  # Monday-Friday
            trading_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return trading_days

def test_random_month():
    """Test predictions for a random month with cached data."""
    print("=" * 70)
    print("Testing MC Dropout + Temperature T=5.0 on Random Month")
    print("=" * 70)
    
    # Get random trading days
    test_dates = get_random_month_dates()
    
    print(f"\nTest Period: {test_dates[0]} to {test_dates[-1]}")
    print(f"Number of days: {len(test_dates)}")
    print("=" * 70)
    
    # Initialize trading system
    trading_system = TrainingSystem()
    
    # Preload data ONCE for entire test period
    # Need 60 days before first date + 10 days buffer after last date
    first_date = datetime.strptime(test_dates[0], "%Y-%m-%d")
    last_date = datetime.strptime(test_dates[-1], "%Y-%m-%d")
    cache_start = (first_date - timedelta(days=60)).strftime("%Y-%m-%d")
    cache_end = (last_date + timedelta(days=10)).strftime("%Y-%m-%d")
    
    print(f"\n📥 Downloading data once for entire period...")
    trading_system.preload_data(cache_start, cache_end)
    
    # Track results
    stock_counts = {}
    confidence_levels = []
    trades = []
    
    print(f"\n📊 Running {len(test_dates)} predictions (using cached data)...")
    print("-" * 70)
    
    for i, test_date in enumerate(test_dates, 1):
        try:
            action, target_stock, top3_choices = trading_system.predict_action(test_date)
            
            top_action, top_stock, top_confidence = top3_choices[0]
            stock_choice = top_stock if top_stock else top_action
            
            # Track stats
            stock_counts[stock_choice] = stock_counts.get(stock_choice, 0) + 1
            confidence_levels.append(top_confidence)
            
            # Calculate 1-day return if it's a stock pick
            if top_stock and i < len(test_dates):  # Need next day for return calculation
                try:
                    current_date = test_date
                    next_date = test_dates[i] if i < len(test_dates) else None
                    
                    if next_date and top_stock in trading_system._cached_data.columns.get_level_values(0):
                        # Get current and next day prices
                        current_slice = trading_system._cached_data.loc[current_date:current_date]
                        next_slice = trading_system._cached_data.loc[next_date:next_date]
                        
                        if not current_slice.empty and not next_slice.empty:
                            buy_price = current_slice[top_stock]['Close'].iloc[0]
                            sell_price = next_slice[top_stock]['Close'].iloc[0]
                            
                            if buy_price > 0 and sell_price > 0:
                                return_pct = (sell_price - buy_price) / buy_price
                                trades.append({
                                    'date': test_date,
                                    'stock': top_stock,
                                    'return': return_pct
                                })
                            else:
                                return_pct = 0.0
                        else:
                            return_pct = 0.0
                    else:
                        return_pct = 0.0
                        
                except Exception as e:
                    return_pct = 0.0
            else:
                return_pct = 0.0
            
            # Display
            return_str = f"{return_pct:+6.2%}" if return_pct != 0.0 else "   N/A "
            print(f"Day {i:2d} ({test_date}): {stock_choice:6s} | Conf: {top_confidence:6.3f} | Return: {return_str}")
            
        except Exception as e:
            print(f"Day {i:2d} ({test_date}): ERROR - {e}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)
    
    print(f"\nConfidence Statistics:")
    avg_conf = sum(confidence_levels) / len(confidence_levels)
    print(f"  Average: {avg_conf:.3f} ({avg_conf*100:.2f}%)")
    print(f"  Min: {min(confidence_levels):.3f} ({min(confidence_levels)*100:.2f}%)")
    print(f"  Max: {max(confidence_levels):.3f} ({max(confidence_levels)*100:.2f}%)")
    
    print(f"\nStock Diversity:")
    unique_stocks = len(stock_counts)
    print(f"  Unique stocks selected: {unique_stocks} out of {len(test_dates)} days")
    
    print(f"\nTop 5 Most Selected:")
    for stock, count in sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = (count / len(test_dates)) * 100
        print(f"  {stock:6s}: {count:2d} days ({pct:5.1f}%)")
    
    # Portfolio performance
    if trades:
        print(f"\n💰 Portfolio Performance:")
        total_return = sum(t['return'] for t in trades)
        avg_return = total_return / len(trades)
        winning_trades = sum(1 for t in trades if t['return'] > 0)
        losing_trades = sum(1 for t in trades if t['return'] < 0)
        
        # Calculate final portfolio value (starting with $10,000)
        portfolio_value = 10000.0
        for trade in trades:
            portfolio_value *= (1 + trade['return'])
        
        profit_loss = portfolio_value - 10000.0
        
        print(f"  Total trades: {len(trades)}")
        print(f"  Winning trades: {winning_trades}")
        print(f"  Losing trades: {losing_trades}")
        print(f"  Win rate: {(winning_trades/len(trades)*100):.1f}%")
        print(f"  Total return: {total_return:+.2%}")
        print(f"  Average return per trade: {avg_return:+.2%}")
        print(f"  Portfolio value (starting $10,000):")
        print(f"    Ending: ${portfolio_value:,.2f}")
        print(f"    Profit/Loss: ${profit_loss:+,.2f}")
    else:
        print(f"\n💰 No trades with calculable returns")
    
    # Assessment
    print("\n" + "=" * 70)
    print("✅ ASSESSMENT")
    print("=" * 70)
    
    if avg_conf < 0.05:
        print(f"✅ Confidence: EXCELLENT (<5%) - Overconfidence fixed!")
    elif avg_conf < 0.10:
        print(f"✅ Confidence: GOOD (<10%)")
    else:
        print(f"⚠️  Confidence: HIGH ({avg_conf*100:.1f}%)")
    
    if unique_stocks >= 10:
        print(f"✅ Diversity: EXCELLENT ({unique_stocks} different stocks)")
    elif unique_stocks >= 5:
        print(f"✅ Diversity: GOOD ({unique_stocks} different stocks)")
    elif unique_stocks <= 2:
        print(f"🚨 Diversity: POOR (only {unique_stocks} stocks - ENPH/NI issue)")
    else:
        print(f"⚠️  Diversity: MODERATE ({unique_stocks} stocks)")
    
    print("=" * 70)

if __name__ == "__main__":
    test_random_month()
