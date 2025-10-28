#!/usr/bin/env python3
"""
Ultra-minimal transformer-based stock trader for S&P 500.
Makes daily stock selection decisions using trained transformer model.

Main CLI interface for making trading predictions.
"""

import argparse
from datetime import datetime

from training_system import TrainingSystem


def main():
    """Command-line interface for the stock trader predictions."""
    parser = argparse.ArgumentParser(description="Transformer Stock Trader - Predictions")
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)', 
                       default=datetime.now().strftime("%Y-%m-%d"))
    
    args = parser.parse_args()
    
    # Initialize the trading system
    trading_system = TrainingSystem()
    
    # Make prediction
    target_stock, top3_choices = trading_system.predict_action(args.date)
    
    print(f"\n🎯 Trading Decision for {args.date}:")
    print(f"Selected Stock: {target_stock}")
    print(f"Top 3 Choices: {[f'{stock} ({prob:.3f})' for stock, prob in top3_choices]}")


if __name__ == "__main__":
    main()