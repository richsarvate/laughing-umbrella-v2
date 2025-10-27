# Feature Request: Fix Model Overfitting

## Problem
Model outputs 90-99% confidence and only picks 2 stocks (NI, ENPH) out of 308 options. This is overconfidence, not good prediction.

## Root Cause
Dropout is disabled during inference (`model.eval()`), making predictions deterministic instead of uncertain.

## Critical Fixes Needed

### 1. Monte Carlo Dropout (PRIORITY 1) ✅ COMPLETED
- Keep model in `train()` mode during prediction
- Run 10-20 forward passes, average results
- **Impact**: Confidence drops from 90-99% to 10-30%
- **Result**: Reduced confidence from 98.9% → 75.9% (23% improvement)

### 2. Temperature Scaling ✅ COMPLETED  
- Add temperature parameter (T=3-5) to soften predictions
- Divide logits by temperature before softmax
- **Impact**: More realistic probability distributions
- **Result**: T=5.0 reduces confidence from 98.9% → 3.1% (97% improvement!)

### 3. Enhanced Regularization
- Increase dropout from 25% to 40%
- Add weight decay (1e-4)
- **Impact**: Better generalization across all stocks

### 4. Ensemble Method
- Train 3 models with different seeds
- Average their predictions
- **Impact**: Natural uncertainty + more diverse picks

## Target Results
- **Before**: 90-99% confidence, only 2 stocks
- **After**: 10-30% confidence, 20+ different stocks

## Implementation
1. Start with MC Dropout (biggest impact)
2. Add temperature scaling
3. Increase regularization
4. Build ensemble

## Files to Modify
- `core/model.py` - MC Dropout, temperature
- `core/trader.py` - Inference changes
- `core/training_system.py` - Regularization