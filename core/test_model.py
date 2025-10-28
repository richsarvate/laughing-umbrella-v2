#!/usr/bin/env python3
"""
Model testing script
"""

import argparse
import os
import torch
from datetime import datetime, timedelta
from training_system import TrainingSystem

def main():
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with minimal data')
    args = parser.parse_args()
    
    if args.quick:
        print("🏁 Starting quick model test...")
    else:
        print("🏁 Starting full model test...")
    
    # Create training system and preload data
    training_system = TrainingSystem()
    
    # Preload data for the test period to avoid repeated downloads
    print("📊 Preloading market data for 2024...")
    training_system.preload_data("2023-09-01", "2024-12-31")
    print("✅ Data preloaded successfully!")
    
    # Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained_stock_trader.pth')
    training_system.model.load_state_dict(torch.load(model_path))
    print("✅ Model loaded successfully!")
    
    # Test prediction functionality
    test_date = "2024-01-02"
    print(f"\n🔮 Testing predictions for {test_date}...")
    
    try:
        # Use predict_action which now uses cached data
        target_stock, top3_choices = training_system.predict_action(test_date)
        
        print(f"\n🎯 Top prediction: {target_stock}")
        print(f"\n📊 Top 3 predictions:")
        for i, (stock, confidence) in enumerate(top3_choices, 1):
            print(f"   {i}. {stock}: {confidence:.4f}")
        
        if not args.quick:
            # Test a few more dates for full test
            additional_dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
            print(f"\n🔄 Testing additional dates...")
            
            for date in additional_dates:
                try:
                    stock, choices = training_system.predict_action(date)
                    print(f"   {date}: {stock} (conf: {choices[0][1]:.4f})")
                except Exception as e:
                    print(f"   {date}: ❌ Error - {str(e)}")
        
    except Exception as e:
        print(f"❌ Failed to generate predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Model test complete!")

if __name__ == "__main__":
    main()