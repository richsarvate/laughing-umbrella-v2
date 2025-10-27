# Stock Trading Transformer V2

A transformer model that learns to pick winning S&P 500 stocks using 60-day price sequences.

## How It Works

**Input**: 60 days of price history for 308 S&P 500 stocks (3 features: normalized price + validity mask + log-normalized volume)  
**Output**: Which stock to buy  
**Training**: Direct profit optimization on 5 years of historical data (2020-2024)

## Key Features

- ✅ **Pure stock selection** - Always picks a stock, no HOLD/CASH options
- ✅ **Stock shuffling** - Prevents memorizing position patterns during training
- ✅ **Validity masking** - Automatically avoids delisted stocks
- ✅ **Volume signal** - Log-normalized volume preserves relative stock size (blue chip vs small cap)
- ✅ **Profit-based loss** - Optimizes actual 5-day returns, not classification
- ✅ **Temperature scaling** - T=1.0 for confident predictions (4.5% on top picks)

## Training

```bash
cd core
python train_v2.py
```

**Training time**: ~6 minutes on CPU (30 epochs)  
**Output**: `models/trained_stock_trader.pth`

## Testing

```bash
cd core  
python portfolio_test.py
```

Tests model over 4 cycles of 5 trading days each.

## Architecture

- **Transformer backbone**: 2 layers, 4 attention heads, 256 hidden dimensions
- **Input**: [60 days, 308 stocks, 3 features] sequences
- **Decision head**: 308 stock outputs (one per valid stock)
- **Loss function**: EnhancedProfitLoss (equal treatment of gains/losses)

## Philosophy

Like GPT learns patterns from text, this model learns market patterns from price sequences. Features: normalized price (trend), validity mask (avoid delisted), log-volume (stock size). The model discovers which combinations predict returns.