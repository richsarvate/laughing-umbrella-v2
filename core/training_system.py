"""
Training system for transformer stock trader.
Handles label generation, model training, and prediction logic.
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_processor import MarketDataProcessor
from model import TransformerStockTrader


class EnhancedProfitLoss(nn.Module):
    """Loss function that directly maximizes returns with asymmetric risk penalties."""
    
    def __init__(self, loss_penalty_factor: float = 2.0, risk_penalty: float = 0.01):
        super().__init__()
        self.loss_penalty_factor = loss_penalty_factor  # Punish losses more than reward gains
        self.risk_penalty = risk_penalty  # Small penalty for cash to encourage risk-taking
        
    def forward(self, predictions: torch.Tensor, future_returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate profit-based loss for model predictions using differentiable approach.
        
        Args:
            predictions: [batch_size, 102] - model logits for all choices
            future_returns: [batch_size, 100] - actual 5-day returns for each stock
        
        Returns:
            loss: Scalar loss value (negative expected return)
        """
        batch_size = predictions.shape[0]
        
        # Convert logits to probabilities (differentiable)
        action_probs = F.softmax(predictions, dim=-1)  # [batch_size, 102]
        
        # Create expanded returns tensor including HOLD and CASH options
        # [batch_size, 102] = [HOLD, CASH, stock_0, stock_1, ..., stock_99]
        expanded_returns = torch.zeros_like(action_probs, device=predictions.device)
        
        # Set returns for each action type
        expanded_returns[:, 0] = 0.0  # HOLD return = 0%
        expanded_returns[:, 1] = -self.risk_penalty  # CASH return = small penalty
        expanded_returns[:, 2:] = future_returns  # Stock returns
        
        # Calculate expected return using probability weighting (differentiable)
        expected_returns = torch.sum(action_probs * expanded_returns, dim=-1)
        
        # Apply asymmetric penalty for negative expected returns
        enhanced_returns = torch.where(
            expected_returns >= 0,
            expected_returns,  # Reward positive expected returns
            self.loss_penalty_factor * expected_returns  # Punish negative expected returns more
        )
        
        # Loss = negative expected return (maximize return = minimize negative return)
        loss = -enhanced_returns.mean()
        
        return loss


class TrainingSystem:
    """Orchestrates training and prediction for the stock trading model."""
    
    def __init__(self):
        self.data_processor = MarketDataProcessor(lookback_days=60)  # Changed to 60 days
        # Use actual number of successfully downloaded stocks
        actual_num_stocks = len(self.data_processor.sp500_tickers)
        self.model = TransformerStockTrader(num_stocks=actual_num_stocks)
        self.current_position = None  # Track current stock holding
        
        # Data cache for multiple predictions
        self._cached_data = None
        self._cache_start = None
        self._cache_end = None
    
    def _mc_dropout_inference(self, input_tensor: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
        """
        Monte Carlo Dropout inference for uncertainty estimation.
        Runs multiple forward passes with dropout enabled and averages the results.
        
        Args:
            input_tensor: Model input tensor
            num_samples: Number of forward passes to average
        Returns:
            Average probabilities across all samples
        """
        self.model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs[0])
        
        # Average across all samples
        avg_probs = torch.stack(predictions).mean(dim=0)
        return avg_probs
    
    def _temperature_inference(self, input_tensor: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
        """
        Temperature-scaled inference for reduced overconfidence.
        
        Args:
            input_tensor: Model input tensor
            temperature: Scaling factor (higher = softer probabilities)
        Returns:
            Temperature-scaled probabilities
        """
        self.model.eval()  # Standard evaluation mode
        
        with torch.no_grad():
            scaled_logits = self.model.forward_with_temperature(input_tensor, temperature=temperature)
            probs = torch.softmax(scaled_logits, dim=1)
        
        return probs[0]
    
    def _combined_inference(self, input_tensor: torch.Tensor, num_mc_samples: int = 20, temperature: float = 5.0) -> torch.Tensor:
        """
        Combined MC Dropout + Temperature Scaling inference.
        Provides both diversity (MC Dropout) and reduced overconfidence (Temperature).
        
        Args:
            input_tensor: Model input tensor
            num_mc_samples: Number of MC Dropout samples
            temperature: Temperature scaling factor
        Returns:
            Combined probabilities
        """
        self.model.train()  # Enable dropout for MC sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_mc_samples):
                # Apply temperature scaling during each MC sample
                scaled_logits = self.model.forward_with_temperature(input_tensor, temperature=temperature)
                probs = torch.softmax(scaled_logits, dim=1)
                predictions.append(probs[0])
        
        # Average across all MC samples
        avg_probs = torch.stack(predictions).mean(dim=0)
        return avg_probs
    
    def preload_data(self, start_date: str, end_date: str):
        """
        Preload and cache market data for a date range.
        Useful when making multiple predictions to avoid repeated downloads.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        print(f"Preloading data from {start_date} to {end_date}...")
        self._cached_data = self.data_processor.download_market_data(start_date, end_date)
        self._cache_start = start_date
        self._cache_end = end_date
        print("✅ Data cached")
    
    def calculate_future_returns(self, raw_market_data, lookahead_days: int = 5) -> np.ndarray:
        """Calculate actual future returns for all stocks for profit-based training."""
        num_days = len(raw_market_data)
        num_stocks = len(self.data_processor.sp500_tickers)
        
        # Initialize returns matrix: [days, stocks]
        future_returns = np.zeros((num_days - lookahead_days, num_stocks))
        
        for i, ticker in enumerate(self.data_processor.sp500_tickers):
            try:
                # Get price series for this stock
                if ticker in raw_market_data.columns.get_level_values(0):
                    close_prices = raw_market_data[ticker]['Close'].values
                    
                    # Calculate forward returns for each day
                    for day in range(len(close_prices) - lookahead_days):
                        current_price = close_prices[day]
                        future_price = close_prices[day + lookahead_days]
                        
                        if current_price > 0 and not np.isnan(current_price) and not np.isnan(future_price):
                            # Calculate percentage return
                            return_pct = (future_price - current_price) / current_price
                            future_returns[day, i] = return_pct
                        else:
                            future_returns[day, i] = 0.0  # No return if invalid prices
                
            except Exception as e:
                print(f"Warning: Could not calculate returns for {ticker}: {e}")
                future_returns[:, i] = 0.0  # Set returns to 0 for problematic stocks
        
        # Clean up extreme values (cap at +/- 50% daily return)
        future_returns = np.clip(future_returns, -0.5, 0.5)
        
        return future_returns
    
    def train_model(self, start_date: str = "2010-01-01", end_date: str = "2024-01-01"):
        """Train the transformer on historical market data using profit optimization with stock shuffling."""
        print("Training transformer stock trader with profit maximization + stock shuffling...")
        print("Using 60-day price sequences (pure sequence learning like GPT)")
        
        # Download and process training data
        raw_market_data = self.data_processor.download_market_data(start_date, end_date)
        price_sequences = self.data_processor.extract_price_sequences(raw_market_data)  # [days, stocks, 1]
        
        # Calculate actual future returns for all stocks
        future_returns = self.calculate_future_returns(raw_market_data)
        
        # Prepare training sequences (60-day windows)
        sequence_length = 60
        training_sequences = []
        training_returns = []
        
        for i in range(sequence_length, len(price_sequences) - 5):  # Leave room for future returns
            sequence = price_sequences[i-sequence_length:i]
            returns = future_returns[i-sequence_length]  # Future returns for this day
            training_sequences.append(sequence)
            training_returns.append(returns)
        
        # Convert to tensors
        X_train = torch.FloatTensor(np.array(training_sequences))  # [N, 60, stocks, 1]
        y_train = torch.FloatTensor(np.array(training_returns))    # [N, stocks]
        
        num_stocks = len(self.data_processor.sp500_tickers)
        print(f"Training on {len(X_train)} sequences with {num_stocks} stocks")
        print(f"Stock shuffling: ENABLED (prevents position memorization)")
        
        # Training loop with profit-based loss and stock shuffling
        optimizer = optim.Adam(self.model.parameters(), lr=8e-5)
        loss_function = EnhancedProfitLoss(loss_penalty_factor=2.0, risk_penalty=0.005)
        
        self.model.train()
        num_epochs = 150
        batch_size = 32
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Shuffle training data each epoch
            indices = torch.randperm(len(X_train))
            
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                batch_indices = indices[batch_start:batch_end]
                
                # Get batch data
                batch_X = X_train[batch_indices]  # [batch, 60, stocks, 1]
                batch_y = y_train[batch_indices]  # [batch, stocks]
                
                # CRITICAL: Shuffle stock order for this batch
                # This prevents the model from learning "position 2 = ENPH = profit"
                stock_shuffle = torch.randperm(num_stocks)
                
                # Apply shuffle to both inputs and outputs
                batch_X_shuffled = batch_X[:, :, stock_shuffle, :]  # Shuffle stocks in input
                batch_y_shuffled = batch_y[:, stock_shuffle]         # Shuffle stocks in returns
                
                optimizer.zero_grad()
                
                # Forward pass with shuffled stocks
                decision_logits = self.model(batch_X_shuffled)
                
                # Calculate profit-based loss on shuffled returns
                loss = loss_function(decision_logits, batch_y_shuffled)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if epoch % 10 == 0:
                avg_loss = np.mean(epoch_losses)
                avg_return = -avg_loss  # Convert loss back to return
                print(f"Epoch {epoch}, Average Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
        
        # Save trained model
        torch.save(self.model.state_dict(), 'trained_stock_trader.pth')
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.data_processor.feature_scaler, f)
        
        final_return = -np.mean(epoch_losses)
        print(f"Model training completed! Final average return: {final_return:.4f} ({final_return*100:.2f}%)")
        print("✅ Stock shuffling applied - model cannot memorize positions!")
    
    def predict_action(self, target_date: str, model_path: str = None, scaler_path: str = None) -> Tuple[str, Optional[str]]:
        """
        Make unified trading decision for a specific date.
        Uses MC Dropout + Temperature Scaling for robust predictions with reduced overconfidence.
        """
        # Use provided paths or default to models directory
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained_stock_trader.pth')
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'feature_scaler.pkl')
            
        # Load trained model
        self.model.load_state_dict(torch.load(model_path))
        with open(scaler_path, 'rb') as f:
            self.data_processor.feature_scaler = pickle.load(f)
        
        # Get recent market data leading up to prediction date (need 60 days now)
        start_date = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Check cache before downloading
        if (self._cached_data is not None and 
            self._cache_start and self._cache_end and
            start_date >= self._cache_start and target_date <= self._cache_end):
            # Use cached data
            raw_market_data = self._cached_data.loc[start_date:target_date]
        else:
            # Download fresh data
            raw_market_data = self.data_processor.download_market_data(start_date, target_date)
        
        # Extract price sequences instead of technical features
        price_sequences = self.data_processor.extract_price_sequences(raw_market_data)
        
        # Use last 60 days as input sequence
        input_sequence = price_sequences[-60:].reshape(1, 60, -1, 1)  # [1, 60, stocks, 1]
        input_tensor = torch.FloatTensor(input_sequence)
        
        # Use combined MC Dropout + Temperature Scaling inference
        decision_probabilities = self._combined_inference(
            input_tensor, 
            num_mc_samples=20, 
            temperature=5.0
        )
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(decision_probabilities, k=3)
        top3_probs = top3_probs.tolist()
        top3_indices = top3_indices.tolist()
        
        predicted_choice = top3_indices[0]
        confidence = top3_probs[0]
        
        # Decode top 3 predictions
        top3_choices = []
        for idx, prob in zip(top3_indices, top3_probs):
            if idx == 0:
                choice_name = "HOLD"
                choice_stock = None
            elif idx == 1:
                choice_name = "CASH"
                choice_stock = None
            else:
                choice_name = "SWITCH"
                choice_stock = self.data_processor.sp500_tickers[idx - 2]
            top3_choices.append((choice_name, choice_stock, prob))
        
        # Decode unified prediction (top choice)
        action_name, target_stock, _ = top3_choices[0]
        
        print(f"MC Dropout + Temperature (T=5.0) - Action: {action_name} (confidence: {confidence:.3f})")
        if target_stock:
            print(f"Model selected: {target_stock}")
        
        return action_name, target_stock, top3_choices