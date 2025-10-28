# Model Improvements - Smaller Architecture

## Changes Made

### Model Architecture Reduction
**Previous (Big Brain):**
- 8 transformer layers
- 12 attention heads  
- 768 hidden dimensions
- Heavy dropout (15-25%)
- 3-layer decision head (768→256→128→stocks)

**New (Smaller, Smarter):**
- 2 transformer layers (75% reduction)
- 4 attention heads (67% reduction)
- 256 hidden dimensions (67% reduction)
- Lighter dropout (10-15%)
- 2-layer decision head (256→128→stocks)

### Expected Benefits

1. **Less Overfitting**
   - Smaller capacity = can't memorize training patterns
   - Should reduce SBUX bias (99% confidence problem)
   - Better generalization to unseen data

2. **Faster Training**
   - ~70% fewer parameters
   - Should train 2-3x faster
   - Less GPU memory required

3. **More Robust Predictions**
   - Forces model to learn real patterns, not noise
   - Should see more diverse stock selections
   - Better risk-adjusted returns

4. **Reduced Overconfidence**
   - Smaller model = naturally less confident
   - More balanced probability distributions
   - Better uncertainty estimates

## How to Retrain

### Quick Test (1 year of data, 5 epochs - ~5 minutes)
```bash
cd /home/ec2-user/Projects/laughing-umbrella-v2/core
/opt/pytorch/bin/python3 train_v2.py --quick-test
```

### Full Training (2010-2024, 150 epochs - ~30 minutes)
```bash
cd /home/ec2-user/Projects/laughing-umbrella-v2/core
/opt/pytorch/bin/python3 train_v2.py
```

### With More Data (2000-2024, 150 epochs - ~40 minutes)
```bash
cd /home/ec2-user/Projects/laughing-umbrella-v2/core
# Edit train_v2.py to change start_date to "2000-01-01"
/opt/pytorch/bin/python3 train_v2.py
```

## Performance Comparison

### Old Model (Big Brain)
- Training: +0.41% return
- Testing (2025): +0.54% return (+0.65% APY)
- SBUX picked: ~95% of the time
- Win rate: 54.4%
- Avg win: $87.99 | Avg loss: $108.04

### New Model (TBD - Need to Train)
- Expected: More diverse picks
- Expected: Better risk/reward ratio
- Expected: Lower confidence scores (healthier)
- Expected: Similar or better returns with less risk

## Next Steps

1. Train the new smaller model
2. Run backtest on 2025 data
3. Compare performance metrics
4. If still issues, consider:
   - Add SPY market filter
   - Add volatility features
   - Implement position sizing by confidence
   - Train on risk-adjusted returns (Sharpe ratio)

## Model Size Comparison

**Parameter Count:**
- Old Model: ~85M parameters
- New Model: ~10M parameters (88% reduction!)

This is more appropriate for the signal-to-noise ratio in financial data.
