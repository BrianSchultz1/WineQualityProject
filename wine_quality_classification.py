import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from ucimlrepo import fetch_ucirepo

# Buscar e carregar a base de dados do UC Irvine Machine Learning Repository
wine_quality = fetch_ucirepo(id=186)

# Dados (como dataframes do pandas)
X = wine_quality.data.features
y = wine_quality.data.targets

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

# Algoritmo 1: Árvore de Decisão
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Algoritmo 2: SVM
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Avaliação dos Modelos


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1


# Resultados Árvore de Decisão
acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(y_test, y_pred_dt)

# Resultados SVM
acc_svm, prec_svm, rec_svm, f1_svm = evaluate_model(y_test, y_pred_svm)

# Organizar resultados em uma tabela
results = [
    ["Decision Tree", acc_dt, prec_dt, rec_dt, f1_dt],
    ["SVM", acc_svm, prec_svm, rec_svm, f1_svm]
]

# Cabeçalhos da tabela
headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

# Exibir tabela formatada
print(tabulate(results, headers=headers, tablefmt="pretty"))
