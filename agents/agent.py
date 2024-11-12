# agents/agent.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Agent(nn.Module, ABC):
    def __init__(self):
        super(Agent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @abstractmethod
    def forward(self, x):
        pass

    def train_model(self, train_loader, criterion, optimizer, epochs=10):
        """
        Common training method for all models.
        Trains for a specified number of epochs.
        """
        self.train()
        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
            average_loss = running_loss / total_samples
            epoch_losses.append(average_loss)
            print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")
        return epoch_losses

    def evaluate(self, test_loader):
        """
        Common evaluation method to calculate model accuracy.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy
