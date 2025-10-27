"""
Training Script V2 - 60-day price sequences with stock shuffling
Fixes position memorization problem (ENPH/NI overfitting)
"""

from training_system import TrainingSystem

if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING TRANSFORMER V2")
    print("=" * 70)
    print("\nNew features:")
    print("  ✅ 60-day price sequences (was 30-day technical indicators)")
    print("  ✅ Stock shuffling (prevents position memorization)")
    print("  ✅ Pure sequence learning (like GPT)")
    print("\n" + "=" * 70)
    
    # Initialize training system
    trainer = TrainingSystem()
    
    # Train on 2010-2024 data
    print("\nStarting training on 2010-01-01 to 2024-01-01...")
    trainer.train_model(start_date="2010-01-01", end_date="2024-01-01")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel saved to: core/trained_stock_trader.pth")
    print("Scaler saved to: core/feature_scaler.pkl")
    print("\nNext steps:")
    print("  1. Run: python test_month.py")
    print("  2. Check for diverse stock picks (not just ENPH/NI)")
    print("  3. Verify portfolio returns")
