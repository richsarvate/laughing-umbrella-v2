# Quick Reference: Diversity Penalty

## The Problem
Model predicted BLK with 99% confidence in 94% of trading cycles despite other stocks having better returns.

## The Solution
Three-component loss function:

### 1. Profit Maximization (existing)
```python
expected_return = sum(probs * returns)
loss_component_1 = -expected_return
```

### 2. Entropy Bonus (NEW)
```python
entropy = -sum(p * log(p))
normalized = entropy / log(num_stocks)
loss_component_2 = -diversity_weight * normalized  # Subtract = reward
```

### 3. Concentration Penalty (NEW)
```python
max_prob = max(probs)
penalty = relu(max_prob - 0.9) ** 2
loss_component_3 = +0.5 * penalty  # Add = penalize
```

## Current Settings
```python
diversity_weight = 0.01  # Tune this: 0.0 = no penalty, 0.1 = very strong
```

## Monitoring
Watch **MaxP** during training:
- 99% = Overfitting (bad)
- 30-50% = Good balance (target)
- 0.4% = Too random (current)

## Tuning Guide
If MaxP too high (>80%): Increase diversity_weight  
If MaxP too low (<10%): Decrease diversity_weight

## Files Changed
- `core/training_system.py` - EnhancedProfitLoss class
- `core/train_v2.py` - Updated docstring

## Expected Impact
**Before**: BLK in 50/53 cycles (94%)  
**After**: Diverse stock picks throughout year
