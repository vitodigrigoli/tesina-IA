# agents/naive_bayes_agent.py
import torch

class NaiveBayesAgent:
    """Implementation of a Multinomial Naive Bayes classifier."""
    def __init__(self, num_features, num_classes):
        """
        Initialize the NaiveBayesAgent with the specified parameters.

        Args:
            num_features (int): Number of features in the dataset.
            num_classes (int): Number of target classes.
        """
        self.num_classes = num_classes
        self.num_features = num_features
        self.class_log_prior = torch.zeros(num_classes)
        self.feature_log_prob = torch.zeros(num_classes, num_features)  # Dimensions: classes x features

    def fit(self, X, y):
        """
        Train the Naive Bayes model on the provided data.

        Args:
            X (torch.Tensor): Input data tensor of shape (num_samples, num_features).
            y (torch.Tensor): Label tensor of shape (num_samples).
        """
        # Ensure the data is of type float
        X = X.float()
        y = y.long()

        # Compute class log priors
        class_counts = torch.bincount(y, minlength=self.num_classes)
        total_count = y.size(0)
        self.class_log_prior = torch.log(class_counts.float() / total_count)

        # Initialize feature counts for each class
        feature_counts = torch.zeros(self.num_classes, self.num_features)

        for c in range(self.num_classes):
            X_c = X[y == c]
            feature_counts[c, :] = X_c.sum(axis=0)

        # Apply Laplace Smoothing
        smoothed_fc = feature_counts + 1  # Laplace Smoothing with alpha=1
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True) + self.num_features

        # Compute log probabilities of features given classes
        self.feature_log_prob = torch.log(smoothed_fc) - torch.log(smoothed_cc)

    def predict_log_proba(self, X):
        """
        Compute the log probabilities for each class given X.

        Args:
            X (torch.Tensor): Input data tensor of shape (num_samples, num_features).

        Returns:
            torch.Tensor: Log probabilities tensor of shape (num_samples, num_classes).
        """
        X = X.float()
        return X @ self.feature_log_prob.t() + self.class_log_prior

    def predict(self, X):
        """
        Predict the classes for the input data X.

        Args:
            X (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Predicted class labels tensor.
        """
        log_probs = self.predict_log_proba(X)
        return torch.argmax(log_probs, dim=1)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        Args:
            X_test (torch.Tensor): Test data.
            y_test (torch.Tensor): Test labels.

        Returns:
            float: Model accuracy.
        """
        y_pred = self.predict(X_test)
        accuracy = (y_pred == y_test).float().mean().item()
        return accuracy
