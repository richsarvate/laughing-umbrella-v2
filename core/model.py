"""
Transformer model architecture for stock trading decisions.
Minimal transformer design using GPT-2 backbone for market pattern recognition.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class TransformerStockTrader(nn.Module):
    """Minimal transformer model for unified stock trading decisions."""
    
    def __init__(self, num_stocks: int = 500, features_per_stock: int = 2, hidden_size: int = 256):
        super().__init__()
        self.num_stocks = num_stocks
        self.features_per_stock = features_per_stock
        
        # Input projection with dropout: [stocks * features] -> hidden_size
        self.input_projection = nn.Sequential(
            nn.Dropout(0.15),  # Input dropout for noisy financial data
            nn.Linear(num_stocks * features_per_stock, hidden_size)
        )
        
        # Minimal transformer config (2 layers, 4 attention heads)
        transformer_config = GPT2Config(
            vocab_size=1,  # Not used for our case
            n_positions=512,
            n_embd=hidden_size,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.2,
            attn_pdrop=0.1,
        )
        self.transformer_backbone = GPT2Model(transformer_config)
        
        # Stock selection head: directly choose best stock
        num_total_choices = num_stocks  # One choice per stock
        self.unified_decision_head = nn.Sequential(
            nn.Dropout(0.2),       # Input dropout to decision head
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),      # Increased dropout
            nn.Linear(64, num_total_choices)
        )
        
    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.
        Args:
            market_features: [batch_size, sequence_length, num_stocks, features_per_stock]
                            where features_per_stock = 2 (price + validity mask)
        Returns:
            decision_logits: [batch_size, num_stocks] - probabilities for each stock
        """
        batch_size, seq_len, num_stocks, features = market_features.shape
        
        # Flatten stock features for each timestep
        flattened_features = market_features.view(batch_size, seq_len, -1)
        
        # Project to hidden dimension
        projected_input = self.input_projection(flattened_features)
        
        # Pass through transformer (use last hidden state)
        transformer_output = self.transformer_backbone(inputs_embeds=projected_input)
        final_hidden_state = transformer_output.last_hidden_state[:, -1, :]  # [batch, hidden]
        
        # Make unified decision: HOLD/CASH or specific stock choice
        decision_logits = self.unified_decision_head(final_hidden_state)
        
        return decision_logits
    
    def forward_with_temperature(self, market_features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with temperature scaling applied to logits.
        Temperature > 1.0 softens the probability distribution (reduces overconfidence).
        
        Args:
            market_features: [batch_size, sequence_length, num_stocks, features_per_stock]
            temperature: Scaling factor for logits (higher = softer probabilities)
        Returns:
            scaled_logits: [batch_size, 502] - temperature-scaled logits
        """
        logits = self.forward(market_features)
        return logits / temperature