#!/usr/bin/env python3
"""
Download and cache market data for training and backtesting.
This script downloads OHLCV data once and saves it locally.
"""

import argparse
import pickle
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from data_processor import MarketDataProcessor


def download_and_save(start_date, end_date, output_name=None):
    """Download market data and save to pickle file."""
    
    print(f"=" * 70)
    print(f"DOWNLOADING MARKET DATA")
    print(f"=" * 70)
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"=" * 70)
    
    # Initialize data processor
    processor = MarketDataProcessor(lookback_days=60)
    
    # Download data
    print("\n📥 Downloading data from Yahoo Finance...")
    raw_data = processor.download_market_data(start_date, end_date)
    
    # Process into sequences
    print("\n🔧 Processing price sequences...")
    price_sequences = processor.extract_price_sequences(raw_data)
    
    # Prepare output filename
    if output_name is None:
        output_name = f"market_data_{start_date}_{end_date}.pkl"
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'data', 
        output_name
    )
    
    # Save data
    print(f"\n💾 Saving data to: {output_path}")
    data_package = {
        'raw_data': raw_data,
        'price_sequences': price_sequences,
        'start_date': start_date,
        'end_date': end_date,
        'tickers': processor.sp500_tickers,
        'download_timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Print summary
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Data saved successfully!")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Price sequences shape: {price_sequences.shape}")
    print(f"   Number of stocks: {len(processor.sp500_tickers)}")
    print(f"   Number of days: {len(price_sequences)}")
    print(f"\n🎯 To use this data in training, update train_v2.py to load from cache.")
    print(f"=" * 70)


def list_cached_data():
    """List all cached data files."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    if not os.path.exists(data_dir):
        print("No data directory found.")
        return
    
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print("No cached data files found.")
        return
    
    print(f"\n📦 Cached Data Files:")
    print(f"=" * 70)
    
    for filename in sorted(pkl_files):
        filepath = os.path.join(data_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Try to load metadata
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            start = data.get('start_date', 'Unknown')
            end = data.get('end_date', 'Unknown')
            timestamp = data.get('download_timestamp', 'Unknown')
            print(f"\n📄 {filename}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Period: {start} to {end}")
            print(f"   Downloaded: {timestamp}")
        except:
            print(f"\n📄 {filename}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   (Could not read metadata)")
    
    print(f"\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Download and cache market data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download training data (2010-2024)
  python3 download_data.py --start 2010-01-01 --end 2024-01-01 --name training_data.pkl
  
  # Download backtest data (2025)
  python3 download_data.py --start 2025-01-01 --end 2025-10-28 --name backtest_2025.pkl
  
  # List cached files
  python3 download_data.py --list
        """
    )
    
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--name', type=str, help='Output filename (default: auto-generated)')
    parser.add_argument('--list', action='store_true', help='List cached data files')
    
    args = parser.parse_args()
    
    if args.list:
        list_cached_data()
        return
    
    if not args.start or not args.end:
        parser.print_help()
        print("\n❌ Error: --start and --end dates are required (or use --list)")
        sys.exit(1)
    
    download_and_save(args.start, args.end, args.name)


if __name__ == "__main__":
    main()
