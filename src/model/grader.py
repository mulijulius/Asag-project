import torch
import torch.nn as nn
from typing import Optional


class GradingModel(nn.Module):
    """
    A PyTorch feedforward neural network for grading predictions.
    
    Supports both regression and classification tasks with optional
    Monte Carlo Dropout for uncertainty estimation.
    
    Architecture:
        Linear(input_dim, hidden_dim) → ReLU → Dropout →
        Linear(hidden_dim, hidden_dim) → ReLU → Dropout →
        Linear(hidden_dim, output_dim)
    """
    
    def __init__(self, input_dim: int = 1152, hidden_dim: int = 128, 
                 output_dim: int = 1, dropout_rate: float = 0.3):
        """
        Initialize the GradingModel.
        
        Args:
            input_dim (int): Dimension of input features. Default: 1152
            hidden_dim (int): Dimension of hidden layers. Default: 128
            output_dim (int): Dimension of output. Default: 1 (for regression/binary classification)
            dropout_rate (float): Dropout probability. Default: 0.3
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self._task_type = "regression"  # Default task type
        self._mc_dropout_enabled = False
        
        # Feedforward layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    @property
    def task_type(self) -> str:
        """Get the current task type (regression or classification)."""
        return self._task_type
    
    @task_type.setter
    def task_type(self, value: str):
        """
        Set the task type.
        
        Args:
            value (str): Either "regression" or "classification"
        
        Raises:
            ValueError: If value is not "regression" or "classification"
        """
        if value not in ("regression", "classification"):
            raise ValueError(f"task_type must be 'regression' or 'classification', got '{value}'")
        self._task_type = value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo Dropout mode.
        
        Sets the model to eval() mode while keeping all Dropout layers active.
        This allows stochastic forward passes for uncertainty estimation via
        multiple predictions with different dropout masks.
        """
        self.eval()
        self._mc_dropout_enabled = True
        
        # Keep dropout layers in training mode to apply stochasticity
        self.dropout1.train()
        self.dropout2.train()
    
    def disable_mc_dropout(self):
        """Disable Monte Carlo Dropout mode.
        
        Reverts to standard eval() mode where Dropout layers are disabled.
        """
        self._mc_dropout_enabled = False
        self.eval()
    
    def is_mc_dropout_enabled(self) -> bool:
        """Check if Monte Carlo Dropout is currently enabled."""
        return self._mc_dropout_enabled


def build_model(config) -> GradingModel:
    """Factory function to build a GradingModel from configuration.
    
    Reads model parameters from the provided config object (typically CFG)
    and returns a ready-to-use GradingModel instance.
    
    Args:
        config: Configuration object with MODEL-related parameters.
                Expected attributes:
                - config.MODEL_INPUT_DIM (int, optional): Input dimension. Default: 1152
                - config.MODEL_HIDDEN_DIM (int, optional): Hidden dimension. Default: 128
                - config.MODEL_OUTPUT_DIM (int, optional): Output dimension. Default: 1
                - config.MODEL_DROPOUT_RATE (float, optional): Dropout rate. Default: 0.3
                - config.MODEL_TASK_TYPE (str, optional): Task type. Default: "regression"
    
    Returns:
        GradingModel: A fully initialized GradingModel instance.
    
    Example:
        >>> from src.config.config import CFG
        >>> model = build_model(CFG)
        >>> print(model)
    """
    # Extract model configuration parameters with sensible defaults
    input_dim = getattr(config, 'MODEL_INPUT_DIM', 1152)
    hidden_dim = getattr(config, 'MODEL_HIDDEN_DIM', 128)
    output_dim = getattr(config, 'MODEL_OUTPUT_DIM', 1)
    dropout_rate = getattr(config, 'MODEL_DROPOUT_RATE', 0.3)
    task_type = getattr(config, 'MODEL_TASK_TYPE', 'regression')
    
    # Instantiate the model
    model = GradingModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate
    )
    
    # Set the task type
    model.task_type = task_type
    
    return model
