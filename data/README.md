# Data Caching System

This directory contains cached market data to speed up training and backtesting.

## Quick Start

### Download Training Data (2010-2024)
```bash
cd /home/ec2-user/Projects/laughing-umbrella-v2
/opt/pytorch/bin/python3 scripts/download_data.py --start 2010-01-01 --end 2024-01-01 --name training_data.pkl
```

### Download Backtest Data (2025)
```bash
/opt/pytorch/bin/python3 scripts/download_data.py --start 2025-01-01 --end 2025-10-28 --name backtest_2025.pkl
```

### List Cached Files
```bash
/opt/pytorch/bin/python3 scripts/download_data.py --list
```

## Benefits

- **Faster iterations**: Download once, train many times
- **No API throttling**: Avoid Yahoo Finance rate limits
- **Reproducibility**: Same data for all experiments
- **Offline training**: Train without internet access
- **Data inspection**: Verify quality before training

## File Structure

```
data/
  training_data.pkl      # 2010-2024 data for training
  backtest_2025.pkl      # 2025 data for testing
  .gitignore            # Ignore large cache files
```

## Data Format

Each `.pkl` file contains:
- `raw_data`: Original OHLCV data from Yahoo Finance
- `price_sequences`: Processed sequences ready for model
- `start_date`, `end_date`: Date range
- `tickers`: List of stock symbols
- `download_timestamp`: When data was downloaded

## Usage in Training

To use cached data, update your training script:

```python
import pickle

# Load cached data
with open('data/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

price_sequences = data['price_sequences']
raw_data = data['raw_data']
# ... continue with training
```

## Tips

- Download separate files for training vs testing
- Re-download periodically to get latest data
- Check file sizes - should be ~50-200 MB depending on date range
- Use descriptive filenames for different experiments
