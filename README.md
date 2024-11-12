README
======

**Tesina IA - Di Grigoli Vito Domenico - 0703357**
-----------------------

Predizione dei Risultati di Partite Sportive utilizzando Modelli di Apprendimento Automatico

* * * * *

**Indice**
----------

-   [Introduzione](#introduzione)
-   [Descrizione dei File](#descrizione-dei-file)
    -   [1\. main.py](#1-mainpy)
    -   [2\. agents/agent.py](#2-agentsagentpy)
    -   [3\. agents/naive_bayes_agent.py](#3-agentsnaive_bayes_agentpy)
    -   [4\. agents/perceptron_agent.py](#4-agentsperceptron_agentpy)
    -   [5\. agents/mlp_agent.py](#5-agentsmlp_agentpy)
    -   [6\. agents/bayesian_network_agent.py](#6-agentsbayesian_network_agentpy)
    -   [7\. DAGtrue_SPORTS.csv](#7-dagtrue_sportscsv)
    -   [8\. trainingData_SPORTS_N_10k.csv](#8-trainingdata_sports_n_10kcsv)
-   [Descrizione del Dataset](#descrizione-del-dataset)
-   [Modelli Implementati](#modelli-implementati)
    -   [1\. Naive Bayes](#1-naive-bayes)
    -   [2\. Percettrone](#2-percettrone)
    -   [3\. Rete Neurale Multistrato (MLP)](#3-rete-neurale-multistrato-mlp)
    -   [4\. Rete Bayesiana](#4-rete-bayesiana)
-   [Risultati Ottenuti](#risultati-ottenuti)
-   [Considerazioni Finali](#considerazioni-finali)
-   [Come Eseguire il Codice](#come-eseguire-il-codice)
-   [Requisiti di Sistema](#requisiti-di-sistema)

* * * * *

**Introduzione**
----------------

Questo progetto si propone di predire i risultati di partite sportive utilizzando vari modelli di apprendimento automatico. Abbiamo implementato e confrontato diversi algoritmi, tra cui Naive Bayes, Percettrone, Rete Neurale Multistrato (MLP) e una Rete Bayesiana. L'obiettivo è valutare le performance di ciascun modello nel predire l'esito delle partite basandosi su dati storici.

* * * * *

**Descrizione dei File**
------------------------

### **1\. main.py**

Script principale che gestisce il caricamento dei dati, il preprocessing, l'addestramento e la valutazione dei modelli. Genera anche grafici utili per l'analisi delle performance.

### **2\. agents/agent.py**

Classe astratta `Agent` che definisce i metodi comuni per i modelli basati su PyTorch. Fornisce implementazioni per l'addestramento e la valutazione dei modelli.

### **3\. agents/naive_bayes_agent.py**

Implementazione del classificatore Naive Bayes Multinomiale. Questa classe non eredita da `Agent` poiché non utilizza PyTorch.

### **4\. agents/perceptron_agent.py**

Implementazione del Percettrone utilizzando PyTorch. Eredita dalla classe `Agent` e definisce la struttura del modello e il metodo `forward`.

### **5\. agents/mlp_agent.py**

Implementazione di una Rete Neurale Multistrato (MLP) utilizzando PyTorch. Eredita dalla classe `Agent` e definisce una rete con un livello nascosto, batch normalization e dropout.

### **6\. agents/bayesian_network_agent.py**

Implementazione di un agente per la Rete Bayesiana utilizzando la libreria `bnlearn`. Gestisce la creazione, l'addestramento e l'inferenza sulla rete bayesiana basata su un DAG predefinito.

### **7\. DAGtrue_SPORTS.csv**

File CSV contenente la struttura del DAG (Directed Acyclic Graph) utilizzato per la Rete Bayesiana. Definisce le dipendenze tra le variabili del dataset.

### **8\. trainingData_SPORTS_N_10k.csv**

Dataset principale contenente i dati storici delle partite sportive. Include varie feature e la variabile target `HDA` (Home Win, Draw, Away Win).

* * * * *

**Descrizione del Dataset**
---------------------------

Il dataset `trainingData_SPORTS_N_10k.csv` contiene 10.000 record di partite sportive con le seguenti colonne:

-   `RDlevel`: Livello di differenza di rango tra le squadre.
-   `HTshots`: Numero di tiri della squadra di casa.
-   `ATshots`: Numero di tiri della squadra ospite.
-   `HTshotOnTarget`: Numero di tiri in porta della squadra di casa.
-   `ATshotsOnTarget`: Numero di tiri in porta della squadra ospite.
-   `possession`: Percentuale di possesso palla della squadra di casa.
-   `HTgoals`: Numero di gol segnati dalla squadra di casa.
-   `ATgoals`: Numero di gol segnati dalla squadra ospite.
-   `HDA`: Risultato della partita (`H`: vittoria in casa, `D`: pareggio, `A`: vittoria in trasferta).

Le variabili sono state categorizzate in intervalli per facilitare l'analisi e l'addestramento dei modelli.

* * * * *

**Modelli Implementati**
------------------------

### **1\. Naive Bayes**

Un semplice classificatore che assume l'indipendenza tra le feature. Implementato senza l'utilizzo di PyTorch.

### **2\. Percettrone**

Un modello lineare di classificazione che utilizza PyTorch. È una rete neurale con un singolo livello lineare.

### **3\. Rete Neurale Multistrato (MLP)**

Un modello più complesso che utilizza PyTorch. Include un livello nascosto, funzioni di attivazione non lineari, batch normalization e dropout per prevenire l'overfitting.

### **4\. Rete Bayesiana**

Modello probabilistico che rappresenta le dipendenze tra le variabili utilizzando un DAG. Implementato utilizzando la libreria `bnlearn`, sfrutta il DAG definito in `DAGtrue_SPORTS.csv`.

* * * * *

**Risultati Ottenuti**
----------------------

Dopo aver eseguito l'addestramento e la valutazione dei modelli, sono stati ottenuti i seguenti risultati:

1.  **Naive Bayes**

    -   **Accuratezza:** 31.50%
    -   **Interpretazione:** L'accuratezza bassa indica che il modello non è in grado di catturare efficacemente le relazioni tra le variabili nel dataset. L'assunzione di indipendenza tra le feature potrebbe non essere valida in questo contesto.
2.  **Percettrone**

    -   **Accuratezza:** 96.20%
    -   **Interpretazione:** Il modello lineare ha performato bene, indicando che esistono relazioni lineari tra le feature e la variabile target.
3.  **Rete Neurale Multistrato (MLP)**

    -   **Accuratezza:** 99.80%
    -   **Report di Classificazione:**
        -   **Precisione, Recall, F1-score:** 1.00 per tutte le classi.
    -   **Interpretazione:** L'MLP ha ottenuto risultati eccellenti, suggerendo che il modello è stato in grado di apprendere efficacemente le relazioni complesse nel dataset.
4.  **Rete Bayesiana**

    -   **Accuratezza:** 99.90%
    -   **Interpretazione:** La rete bayesiana ha raggiunto la massima accuratezza tra i modelli testati, indicando che la rappresentazione delle dipendenze tramite il DAG è stata efficace.

* * * * *

**Considerazioni Finali**
-------------------------

-   **Elevata Accuratezza dei Modelli:** Sia l'MLP che la Rete Bayesiana hanno raggiunto accuratezze molto elevate sul set di test. Ciò potrebbe indicare che i modelli sono estremamente efficaci o che potrebbe esserci un overfitting.

-   **Possibile Overfitting:** È importante verificare che i modelli non abbiano memorizzato i dati di training. Tecniche come la cross-validation potrebbero essere utilizzate per valutare meglio la capacità di generalizzazione.

-   **Bassa Performance del Naive Bayes:** La bassa accuratezza del Naive Bayes suggerisce che le assunzioni alla base del modello non sono soddisfatte dal dataset. Le feature potrebbero essere correlate e richiedere modelli più complessi.

-   **Importanza del Preprocessing:** Un corretto preprocessing dei dati, come la codifica e la normalizzazione, è stato fondamentale per il successo dei modelli basati su reti neurali.

-   **Analisi delle Feature:** Ulteriori analisi potrebbero essere svolte per identificare le feature più influenti e comprendere meglio le dinamiche che portano ai risultati delle partite.