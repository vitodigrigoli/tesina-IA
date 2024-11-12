# main.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from agents.naive_bayes_agent import NaiveBayesAgent
from agents.perceptron_agent import PerceptronAgent
from agents.mlp_agent import MLPAgent
from agents.bayesian_network_agent import BayesianNetworkAgent
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Imposta il seme random per la riproducibilità
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funzione per caricare e preprocessare i dati
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    if 'HDA' not in data.columns:
        raise ValueError("La colonna 'HDA' non è presente nel dataset.")
    
    features = data.drop(columns=['HDA'])
    target = data['HDA']
    
    return features, target

# Caricamento dei dati
features, target = load_data('trainingData_SPORTS_N_10k.csv')

# Divisione dei dati in training e test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=42)

# Codifica dell'etichetta target
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(target_train)
y_test_encoded = target_encoder.transform(target_test)

# Codifica delle feature per i modelli neurali
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features_train = encoder.fit_transform(features_train)
encoded_features_test = encoder.transform(features_test)

# Normalizzazione dei dati per i modelli neurali
scaler = StandardScaler()
X_train = scaler.fit_transform(encoded_features_train)
X_test = scaler.transform(encoded_features_test)

# Conversione dei dati in tensori per l'uso con PyTorch DataLoader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Inizializzazione e addestramento dei modelli
print("\n=================== Naive Bayes ===================")
# 1. Naive Bayes
print("Addestramento del modello Naive Bayes...")
naive_bayes = NaiveBayesAgent(num_features=X_train.shape[1], num_classes=len(np.unique(y_train_encoded)))
naive_bayes.fit(X_train_tensor, y_train_tensor)
print("Addestramento completato.")

# 2. Percettrone
print("\n=================== Percettrone ===================")
print("Addestramento del modello Percettrone...")
perceptron = PerceptronAgent(input_size=X_train.shape[1], num_classes=len(np.unique(y_train_encoded)))
perceptron.to(device)
optimizer_perceptron = torch.optim.SGD(perceptron.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
perceptron_losses = perceptron.train_model(train_loader, criterion, optimizer_perceptron, epochs=10)
print("Addestramento completato.")

# 3. Rete Neurale (MLP)
print("\n=================== MLP (Rete Neurale Multistrato) ===================")
print("Addestramento del modello MLP...")
mlp = MLPAgent(input_size=X_train.shape[1], hidden_size=64, num_classes=len(np.unique(y_train_encoded)))
mlp.to(device)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
mlp_losses = mlp.train_model(train_loader, criterion, optimizer_mlp, epochs=10)
print("Addestramento completato.")

# 4. Rete Bayesiana con bnlearn
print("\n=================== Rete Bayesiana ===================")
print("Addestramento del modello di Rete Bayesiana...")
# Usa le feature originali per la rete bayesiana
train_data_with_target = features_train.copy()
train_data_with_target['HDA'] = target_train

# Assicurati che tutte le colonne siano di tipo 'category'
for col in train_data_with_target.columns:
    train_data_with_target[col] = train_data_with_target[col].astype('category')

# Prepara il test set per la rete bayesiana
X_test_df = features_test.copy()
for col in X_test_df.columns:
    X_test_df[col] = X_test_df[col].astype('category')

# Rimuovi 'HDA' da X_test_df se presente
if 'HDA' in X_test_df.columns:
    X_test_df = X_test_df.drop(columns=['HDA'])

bayesian_network = BayesianNetworkAgent('DAGtrue_SPORTS.csv', target_variable='HDA')
bayesian_network.train(train_data_with_target)
print("Addestramento completato.")

# Valutazione di ciascun modello
print("\n=================== Valutazione dei Modelli ===================")
# Naive Bayes
print("\nValutazione del modello Naive Bayes...")
accuracy_nb = naive_bayes.evaluate(X_test_tensor, y_test_tensor)
print(f"Accuratezza del modello Naive Bayes: {accuracy_nb:.4f}")

# Percettrone
print("\nValutazione del modello Percettrone...")
accuracy_perceptron = perceptron.evaluate(test_loader)
print(f"Accuratezza del modello Percettrone: {accuracy_perceptron:.4f}")

# MLP (Rete Neurale Multistrato)
print("\nValutazione del modello MLP...")
accuracy_mlp = mlp.evaluate(test_loader)
print(f"Accuratezza del modello MLP: {accuracy_mlp:.4f}")

# Rete Bayesiana
print("\nValutazione del modello di Rete Bayesiana...")
accuracy_bn, y_pred_bn = bayesian_network.evaluate(X_test_df, target_test)
print(f"Accuratezza del modello di Rete Bayesiana: {accuracy_bn:.4f}")

# Calcolo di metriche aggiuntive per il MLP
print("\n=================== Report di Classificazione per MLP ===================")
mlp.eval()
y_pred_mlp = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred_mlp.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

print(classification_report(y_true, y_pred_mlp, target_names=target_encoder.classes_))

# Matrice di Confusione per MLP
cm = confusion_matrix(y_true, y_pred_mlp)
cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Valori Reali')
plt.xlabel('Valori Predetti')
plt.title('Matrice di Confusione per MLP')
plt.show()

# Grafico della perdita di addestramento nel tempo per MLP
plt.figure()
plt.plot(range(1, len(mlp_losses) + 1), mlp_losses, marker='o')
plt.title('Perdita di Addestramento per MLP nel Tempo')
plt.xlabel('Epoca')
plt.ylabel('Perdita')
plt.grid(True)
plt.show()

# Grafico della perdita di addestramento per Percettrone
plt.figure()
plt.plot(range(1, len(perceptron_losses) + 1), perceptron_losses, marker='o', color='red')
plt.title('Perdita di Addestramento per Percettrone nel Tempo')
plt.xlabel('Epoca')
plt.ylabel('Perdita')
plt.grid(True)
plt.show()
