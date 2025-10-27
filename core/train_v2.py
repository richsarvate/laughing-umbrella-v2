"""
Training Script V2 - 60-day price sequences with stock shuffling and validity masking
Fixes position memorization problem and delisted stock selection
"""

import argparse
from training_system import TrainingSystem

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Stock Trading Transformer V2")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test (5 epochs, 1 year of data)")
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.quick_test:
        epochs = 30
        start_date = "2020-01-01"
        end_date = "2024-01-01"
        mode_name = "MEDIUM TRAINING"
        print("=" * 50)
        print(f"TRAINING TRANSFORMER V2 - {mode_name}")
        print("=" * 50)
        print("\nMedium training mode:")
        print("  ⚡ 30 epochs (vs 150 full)")
        print("  📅 2020-2024 data (5 years vs 14 years)")
        print("  🔧 MacBook-friendly training time (~1-2 minutes)")
        print("\n" + "=" * 50)
    else:
        epochs = 150
        start_date = "2010-01-01" 
        end_date = "2024-01-01"
        mode_name = "FULL TRAINING"
        print("=" * 70)
        print(f"TRAINING TRANSFORMER V2 - {mode_name}")
        print("=" * 70)
        print("\nNew features:")
        print("  ✅ 60-day price sequences (was 30-day technical indicators)")
        print("  ✅ Stock shuffling (prevents position memorization)")
        print("  ✅ Validity masking (prevents delisted stock selection)")
        print("  ✅ Pure sequence learning (like GPT)")
        print("\n" + "=" * 70)
    
    # Initialize training system
    trainer = TrainingSystem()
    
    # Train with specified parameters
    print(f"\nStarting training on {start_date} to {end_date}...")
    trainer.train_model(start_date=start_date, end_date=end_date, epochs=epochs)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel saved to: core/trained_stock_trader.pth")
    print("\nNext steps:")
    print("  1. Run: python test_month.py")
    print("  2. Check for diverse stock picks (not just ENPH/NI)")
    print("  3. Verify portfolio returns")
