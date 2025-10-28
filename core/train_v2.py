"""
Training Script V2 - 60-day price sequences with stock shuffling, validity masking, and DIVERSITY PENALTY
Fixes position memorization problem, delisted stock selection, AND BLK overfitting!
NEW: Diversity penalty prevents model from putting 99% confidence on a single stock
"""

import argparse
from training_system import TrainingSystem

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Stock Trading Transformer V2")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test (30 epochs, 5 years of data)")
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.quick_test:
        epochs = 5
        start_date = "2023-01-01"
        end_date = "2024-01-01"
        mode_name = "MEDIUM TRAINING"
        print("=" * 50)
        print(f"TRAINING TRANSFORMER V2 - {mode_name}")
        print("=" * 50)
        print("\nMedium training mode:")
        print("  ⚡ 5 epochs (vs 150 full)")
        print("  📅 2023-2024 data (1 years vs 14 years)")
        print("\n" + "=" * 50)
    else:
        epochs = 150
        start_date = "2000-01-01" 
        end_date = "2024-01-01"
        mode_name = "FULL TRAINING (2000-2024)"
        print("=" * 70)
        print(f"TRAINING TRANSFORMER V2 - {mode_name}")
        print("=" * 70)
        print(f"\n📅 Training Period: {start_date} to {end_date} (24 years)")
        print(f"⚡ Epochs: {epochs}")
        print(f"🚀 GPU Acceleration: ENABLED")
    
    # Initialize training system
    trainer = TrainingSystem()
    
    # Train with specified parameters
    print(f"\nStarting training on {start_date} to {end_date}...")
    trainer.train_model(start_date=start_date, end_date=end_date, epochs=epochs)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel saved to: core/trained_stock_trader.pth")
