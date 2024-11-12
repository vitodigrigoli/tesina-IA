# agents/bayesian_network_agent.py
import bnlearn as bn
import pandas as pd
from sklearn.metrics import accuracy_score
import logging
import sys
import os

class BayesianNetworkAgent:
    def __init__(self, structure_file, target_variable):
        """
        Inizializza l'agente della rete bayesiana.

        Args:
            structure_file (str): Percorso al file CSV contenente la struttura del DAG.
            target_variable (str): Nome della variabile target da prevedere.
        """
        # Imposta il logger di bnlearn per mostrare solo errori
        logging.getLogger('bnlearn').setLevel(logging.ERROR)
        
        # Importa il DAG direttamente dal file CSV usando bnlearn
        dag_data = pd.read_csv(structure_file)
        # Elimina eventuali spazi bianchi dai nomi delle variabili e crea gli archi
        edges = [(row['Variable 1'].strip(), row['Variable 2'].strip()) for _, row in dag_data.iterrows()]
        
        # Crea il DAG usando bnlearn
        self.model = bn.make_DAG(edges)
        
        # Recupera i nomi delle variabili
        self.model_variables = set([edge[0] for edge in edges] + [edge[1] for edge in edges])
        print(f"Variabili nel DAG: {self.model_variables}")
        
        # Imposta la variabile target
        self.target_variable = target_variable.strip()

        if self.target_variable not in self.model_variables:
            raise ValueError(f"La variabile target '{self.target_variable}' non esiste nel DAG.")
    
    def train(self, train_data):
        """
        Addestra il modello di rete bayesiana sui dati forniti.

        Args:
            train_data (pd.DataFrame): Dataset di addestramento contenente le variabili del DAG.
        """
        # Rinomina le colonne del dataset di addestramento per corrispondenza con le variabili nel DAG
        train_data = train_data.rename(columns=lambda x: x.strip())
        
        # Mantieni solo le colonne che sono variabili nel DAG
        columns_to_keep = [col for col in train_data.columns if col in self.model_variables]
        train_data = train_data[columns_to_keep]

        # Assicurati che tutte le variabili siano categoriche
        for col in train_data.columns:
            if not pd.api.types.is_categorical_dtype(train_data[col]):
                train_data[col] = train_data[col].astype('category')

        # Addestra il modello, sopprimendo i warning
        try:
            self.model = bn.parameter_learning.fit(self.model, train_data, verbose=0)
        except Exception as e:
            print(f"Errore durante l'addestramento del modello bayesiano: {e}")
            raise e  # Rilancia l'eccezione per gestirla a livello superiore
    
    def predict(self, X_test):
        """
        Predice le classi per i dati di input X_test.

        Args:
            X_test (pd.DataFrame): Dataset di test contenente le variabili del DAG.

        Returns:
            list: Lista delle predizioni.
        """
        # Rinomina le colonne di X_test per garantire corrispondenza con il DAG
        X_test = X_test.rename(columns=lambda x: x.strip())
        columns_to_keep = [col for col in X_test.columns if col in self.model_variables]
        X_test = X_test[columns_to_keep]
        
        y_pred = []
        for _, row in X_test.iterrows():
            # Crea l'evidenza solo con variabili presenti e non null nel DAG, escludendo la variabile target
            evidence = {
                k: v for k, v in row.to_dict().items()
                if k in self.model_variables and pd.notnull(v) and k != self.target_variable
            }
            
            # Se l'evidenza è vuota, salta la predizione
            if not evidence:
                y_pred.append(None)
                continue

            if self.target_variable:
                try:
                    # Sopprimi l'output di bnlearn durante l'inferenza
                    with open(os.devnull, 'w') as devnull:
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = devnull
                        sys.stderr = devnull
                        try:
                            query_result = bn.inference.fit(
                                self.model,
                                variables=[self.target_variable],
                                evidence=evidence,
                                verbose=0
                            )
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                    # Ottieni la classe con la probabilità più alta
                    max_prob_idx = query_result.df['p'].idxmax()
                    max_prob_class = query_result.df.loc[max_prob_idx][self.target_variable]
                    y_pred.append(max_prob_class)
                except Exception as e:
                    print(f"Errore nell'inferenza: {e}")
                    y_pred.append(None)
            else:
                print("Errore: Variabile target non definita.")
                y_pred.append(None)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Valuta il modello sui dati di test.

        Args:
            X_test (pd.DataFrame): Dati di test.
            y_test (pd.Series or list): Etichette di test.

        Returns:
            tuple: Accuratezza del modello e lista delle predizioni.
        """
        y_pred = self.predict(X_test)
        
        # Calcola il numero di predizioni valide
        num_valid_predictions = sum(1 for pred in y_pred if pred is not None)
        total_predictions = len(y_pred)
        print(f"Predizioni valide: {num_valid_predictions}/{total_predictions}")
        
        # Filtra predizioni valide
        valid_predictions = [(y, pred) for y, pred in zip(y_test, y_pred) if pred is not None]
        
        if not valid_predictions:
            print("Nessuna predizione valida.")
            return float('nan'), y_pred
        
        y_test_filtered, y_pred_filtered = zip(*valid_predictions)
        accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
        return accuracy, y_pred
