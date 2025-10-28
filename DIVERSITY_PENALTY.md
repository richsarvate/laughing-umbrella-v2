# Diversity Penalty Implementation

## Problem: BLK Overfitting

The model was consistently predicting BLK (BlackRock) with 95-99% confidence in almost every trading cycle, despite:
- Stock shuffling during training
- Different market conditions
- Other stocks having similar or better mean returns (e.g., AAPL: 0.607%, MSFT: 0.273%)

**Root Cause**: The model learned to recognize BLK's specific price patterns, volatility characteristics, and price levels, even when shuffled. BLK's patterns are so distinctive that the model could identify it regardless of position.

## Solution: Multi-Component Diversity Penalty

### 1. Entropy Bonus
Rewards the model for spreading predictions across multiple stocks:

```python
# Calculate entropy of probability distribution
entropy = -sum(p * log(p))

# Normalize to [0, 1] range
max_entropy = log(num_stocks)
normalized_entropy = entropy / max_entropy

# Add as bonus (subtract from loss)
diversity_bonus = normalized_entropy.mean()
```

**Effect**: 
- High entropy (uniform distribution) = higher reward
- Low entropy (all probability on one stock) = lower reward

### 2. Concentration Penalty
Specifically penalizes when the model puts >90% confidence on a single stock:

```python
# Find maximum probability across stocks
max_prob = max(action_probs, dim=-1)

# Quadratic penalty above 90% threshold
concentration_penalty = relu(max_prob - 0.9) ** 2
```

**Effect**:
- <90% confidence: No penalty
- 90-95% confidence: Small penalty
- 95-99% confidence: Large penalty (quadratic growth)

### 3. Combined Loss Function

```python
loss = -expected_return                    # Maximize profit
       - (diversity_weight * diversity_bonus)  # Reward spreading
       + (0.5 * concentration_penalty)      # Penalize concentration
```

## Tuning the Diversity Weight

The `diversity_weight` parameter controls how aggressively to enforce diversity:

| Weight | Effect | MaxP* | Behavior |
|--------|--------|-------|----------|
| 0.0    | No penalty | 99% | Overfits to BLK |
| 0.01   | Light penalty | 30-50% | Balanced (RECOMMENDED) |
| 0.1    | Strong penalty | 0.4% | Too random |
| 1.0    | Extreme penalty | 0.3% | Nearly uniform distribution |

*MaxP = Average maximum probability in predictions

## Current Configuration

```python
loss_function = EnhancedProfitLoss(
    loss_penalty_factor=1.0,  # Equal treatment of gains/losses
    diversity_weight=0.01     # Light diversity penalty
)
```

## Expected Results

With diversity penalty enabled:

### Before (diversity_weight=0.0):
```
Cycle 1: BLK (98.9%), APTV (0.15%), DOV (0.13%)
Cycle 2: BLK (99.7%), MRO (0.22%), APTV (0.22%)
Cycle 3: BLK (98.9%), DUK (0.13%), APTV (0.10%)
...
Result: BLK appears in 50/53 cycles
```

### After (diversity_weight=0.01):
```
Cycle 1: BLK (35%), AAPL (22%), MSFT (15%)
Cycle 2: NVDA (28%), BLK (25%), AVGO (18%)
Cycle 3: AAPL (32%), GOOGL (20%), BLK (18%)
...
Result: Multiple stocks, no single dominant choice
```

## Training Metrics

During training, monitor the **MaxP** metric:

```
Epoch   0/150 │ Return: +10.26% │ Best: +10.26% │ MaxP:  45.2% │ Time: 1.8s
Epoch  50/150 │ Return: +11.83% │ Best: +12.01% │ MaxP:  38.5% │ Time: 1.7s
Epoch 100/150 │ Return: +12.45% │ Best: +12.45% │ MaxP:  35.8% │ Time: 1.7s
```

**Target MaxP**: 30-50% (confident but not overconfident)

## Implementation Notes

1. **Differentiable**: All components use differentiable operations (softmax, log, relu)
2. **Batch-aware**: Applied per-sample then averaged across batch
3. **GPU-compatible**: All operations work on CUDA tensors
4. **Stock-agnostic**: Penalty doesn't favor any specific stock

## Future Improvements

1. **Adaptive weight**: Start with strong penalty, decrease over training
2. **Stock-specific penalties**: Track which stocks are over-predicted
3. **Temporal diversity**: Penalize picking same stock multiple days in a row
4. **Sector diversity**: Encourage picking stocks from different sectors

## References

- Entropy-based regularization: [Pereyra et al., 2017](https://arxiv.org/abs/1701.06548)
- Label smoothing: [Szegedy et al., 2016](https://arxiv.org/abs/1512.00567)
- Temperature scaling: [Guo et al., 2017](https://arxiv.org/abs/1706.04599)
