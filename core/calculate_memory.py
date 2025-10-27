"""
Calculate memory and training time for different sequence lengths.
"""

import torch

# Current architecture
num_stocks = 308
hidden_size = 256
batch_size = 32  # typical batch size

print("=" * 70)
print("MEMORY ANALYSIS FOR DIFFERENT SEQUENCE LENGTHS")
print("=" * 70)

sequence_lengths = [30, 60, 90, 120, 180, 252]  # days

for seq_len in sequence_lengths:
    print(f"\n{'='*70}")
    print(f"Sequence Length: {seq_len} days")
    print(f"{'='*70}")
    
    # Input tensor size
    # Current: [batch, seq_len, num_stocks, 3_features]
    # New: [batch, seq_len, num_stocks, 1_feature] or just [batch, seq_len, num_stocks]
    
    current_input = batch_size * seq_len * num_stocks * 3 * 4  # 4 bytes per float32
    new_input = batch_size * seq_len * num_stocks * 1 * 4
    
    print(f"\nInput Tensor Size:")
    print(f"  Current (3 features): {current_input / 1024 / 1024:.2f} MB")
    print(f"  New (1 feature):      {new_input / 1024 / 1024:.2f} MB")
    
    # Projected tensor (after input_projection layer)
    # [batch, seq_len, hidden_size]
    projected = batch_size * seq_len * hidden_size * 4
    print(f"\nProjected Tensor: {projected / 1024 / 1024:.2f} MB")
    
    # Transformer attention memory (quadratic in sequence length!)
    # Self-attention: [batch, n_heads, seq_len, seq_len] per layer
    n_heads = 4
    n_layers = 2
    attention_per_layer = batch_size * n_heads * seq_len * seq_len * 4
    total_attention = attention_per_layer * n_layers
    
    print(f"\nAttention Memory (CRITICAL):")
    print(f"  Per layer: {attention_per_layer / 1024 / 1024:.2f} MB")
    print(f"  Total ({n_layers} layers): {total_attention / 1024 / 1024:.2f} MB")
    
    # Total estimated memory (rough)
    total_memory = current_input + projected + total_attention + 500  # +500 for model params
    print(f"\nEstimated Total: {total_memory / 1024 / 1024:.2f} MB")
    
    # Relative training time (quadratic due to attention)
    relative_time = (seq_len / 30) ** 2
    print(f"\nRelative Training Time: {relative_time:.1f}x slower than 30 days")
    
    if seq_len == 30:
        print("  ✅ Current baseline")
    elif seq_len <= 90:
        print("  ✅ Should be fine")
    elif seq_len <= 120:
        print("  ⚠️  Slower but manageable")
    else:
        print("  🚨 May be too slow")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
Based on transformer attention complexity (O(n²)):

• 30 days:  Baseline (current)
• 60 days:  4x slower, but very manageable
• 90 days:  9x slower, still reasonable (RECOMMENDED)
• 120 days: 16x slower, getting slow
• 252 days: 71x slower, probably too slow

Sweet spot: 60-90 days
- Enough history to capture trends
- Not too slow for experimentation
- Attention memory manageable
""")
