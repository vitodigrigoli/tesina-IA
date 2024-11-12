# agents/mlp_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent import Agent

class MLPAgent(Agent):
    """Implementation of a Multilayer Perceptron for classification."""

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the MLPAgent with the specified parameters.

        Args:
            input_size (int): Number of input units.
            hidden_size (int): Number of units in the hidden layer.
            num_classes (int): Number of classes for classification.
        """
        super(MLPAgent, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(hidden_size, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the layers with appropriate methods."""
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        """
        Perform the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)  # Flatten the input if multidimensional
        x = x.to(self.device)      # Ensure the input is on the correct device
        x = self.hidden(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x  # Return the raw logits
