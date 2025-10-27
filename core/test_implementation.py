#!/usr/bin/env python3
"""
Simple test to verify MC Dropout + Temperature T=5.0 implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_system import TrainingSystem

def test_implementation():
    """Test that MC Dropout + Temperature is working."""
    print("=" * 70)
    print("Testing MC Dropout + Temperature T=5.0 Implementation")
    print("=" * 70)
    
    trading_system = TrainingSystem()
    
    # Test on a single date
    test_date = "2025-09-15"
    
    print(f"\nRunning prediction for {test_date}...")
    print("-" * 70)
    
    try:
        action, target_stock, top3_choices = trading_system.predict_action(test_date)
        
        print("\nResults:")
        print(f"  Action: {action}")
        print(f"  Target Stock: {target_stock if target_stock else 'N/A'}")
        print(f"\n  Top 3 Choices:")
        for i, (act, stk, conf) in enumerate(top3_choices, 1):
            stock_display = stk if stk else act
            print(f"    #{i}: {stock_display:8s} - {conf:.4f} ({conf*100:.2f}%)")
        
        # Check confidence level
        top_confidence = top3_choices[0][2]
        print(f"\n{'='*70}")
        if top_confidence < 0.05:
            print("✅ SUCCESS: Confidence is low (<5%), overconfidence fixed!")
        elif top_confidence < 0.10:
            print("✅ GOOD: Confidence is reasonable (<10%)")
        else:
            print(f"⚠️  WARNING: Confidence still high ({top_confidence*100:.1f}%)")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_implementation()
