# agents/naive_bayes_agent.py
import bnlearn as bn
import pandas as pd
from sklearn.metrics import accuracy_score

class NaiveBayesAgent:
    """Implementazione di un classificatore Naive Bayes utilizzando bnlearn."""
    def __init__(self, target_variable, feature_names):
        """
        Inizializza il NaiveBayesAgent con la variabile target e le feature.

        Args:
            target_variable (str): Nome della variabile target.
            feature_names (list): Lista dei nomi delle feature.
        """
        self.target_variable = target_variable
        self.feature_names = feature_names

        # Creazione del DAG per Naive Bayes
        edges = []
        for feature in self.feature_names:
            edges.append((self.target_variable, feature))
        self.model = bn.make_DAG(edges)

    def train(self, train_data):
        """
        Addestra il modello Naive Bayes sui dati forniti.

        Args:
            train_data (pd.DataFrame): Dataset di addestramento contenente la variabile target e le feature.
        """
        # Assicurati che tutte le colonne siano di tipo 'category'
        for col in train_data.columns:
            if not pd.api.types.is_categorical_dtype(train_data[col]):
                train_data[col] = train_data[col].astype('category')

        # Addestra il modello
        self.model = bn.parameter_learning.fit(self.model, train_data)

    def predict(self, X_test):
        """
        Predice le classi per i dati di input X_test.

        Args:
            X_test (pd.DataFrame): Dataset di test contenente le feature.

        Returns:
            list: Lista delle predizioni.
        """
        y_pred = []
        for _, row in X_test.iterrows():
            evidence = row.to_dict()
            # Esegui l'inferenza per trovare la classe con la probabilità massima
            query_result = bn.inference.fit(
                self.model,
                variables=[self.target_variable],
                evidence=evidence,
                verbose=0
            )
            max_prob_idx = query_result.df['p'].idxmax()
            max_prob_class = query_result.df.loc[max_prob_idx][self.target_variable]
            y_pred.append(max_prob_class)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Valuta il modello sui dati di test.

        Args:
            X_test (pd.DataFrame): Dati di test.
            y_test (pd.Series or list): Etichette di test.

        Returns:
            float: Accuratezza del modello.
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred
