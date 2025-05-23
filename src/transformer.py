import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, hidden_dim=256, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input dimension (features per frame)
            nhead=num_heads,    # Number of attention heads
            dim_feedforward=hidden_dim,  # Feedforward hidden layer size
            dropout=dropout
        )
        
        # Stacked transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        
        # Classifier head
        self.fc = nn.Linear(input_dim, num_classes)  # Final layer to output class probabilities
    
    def forward(self, x):
        """
        x: (batch_size, time_steps, features)
        """
        # Transformer expects input of shape (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Shape: (time_steps, batch_size, features)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # We take the output of the last time step (or average pooling across time)
        # For simplicity, let's use the last time step output (as a representation of the sequence)
        x = transformer_out[-1, :, :]  # Shape: (batch_size, features)
        
        # Classifier head to predict the class
        x = self.fc(x)
        return x


class EnhancedTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=6, hidden_dim=512, dropout=0.2):
        super(EnhancedTransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Transformer encoder layer with more depth, hidden layers, and dropout
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input dimension (features per frame)
            nhead=num_heads,    # Number of attention heads
            dim_feedforward=hidden_dim,  # Feedforward hidden layer size
            dropout=dropout
        )
        
        # Stacked transformer encoder with more layers (increased depth)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        
        # Use global average pooling instead of just the last time step output
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling across the time dimension
        
        # Classifier head
        self.fc = nn.Linear(input_dim, num_classes)  # Final layer to output class probabilities
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization for better training
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        x: (batch_size, time_steps, features)
        """
        # Transformer expects input of shape (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Shape: (time_steps, batch_size, features)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling (across time_steps)
        x = transformer_out.mean(dim=0)  # Shape: (batch_size, features)
        
        # Alternatively, we could also try adaptive pooling
        # x = self.pooling(transformer_out.permute(1, 2, 0)).squeeze(-1)  # Apply global average pooling
        
        # Layer normalization for better stability
        x = self.layer_norm(x)
        
        # Dropout to regularize the output
        x = self.dropout(x)
        
        # Classifier head to predict the class
        x = self.fc(x)
        return x



        