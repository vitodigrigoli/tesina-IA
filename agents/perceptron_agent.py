# agents/perceptron_agent.py
import torch
import torch.nn as nn
from agents.agent import Agent

class PerceptronAgent(Agent):
    """Implementation of a simple perceptron for classification."""

    def __init__(self, input_size, num_classes):
        """
        Initialize the PerceptronAgent.

        Args:
            input_size (int): Number of input features.
            num_classes (int): Number of output classes.
        """
        super(PerceptronAgent, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        Perform the forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)  # Flatten the input if necessary
        x = x.to(self.device)      # Move input to the correct device
        logits = self.linear(x)
        return logits  # Return the raw logits
