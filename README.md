# Projeto Titanic Machine Learning - Kaggle

Bem-vindo ao projeto **Titanic Machine Learning**. Este guia tem como objetivo explicar os processos da criação do codigo e instalação

## Explicando
### 1. Importar bibliotecas
```python
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split, GridSearchCV
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
````
### 2. Carregar os dados
```python
  train = pd.read_csv('train.csv')
  test = pd.read_csv('test.csv')
```
### 3. Tratar valores nulos e converter variáveis
```python
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

label = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train[col] = label.fit_transform(train[col])
    test[col] = label.transform(test[col])
```
### 4. Selecionar colunas relevantes
```python
  features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
  X = train[features]
  y = train['Survived']
```
### 5. Dividir em treino e validação
```python
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 6. Escalar os dados
```python
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
```
### 7. Criar e otimizar o modelo
```python
  param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
  }
  
  logreg = LogisticRegression(max_iter=1000)
  grid = GridSearchCV(logreg, param_grid, cv=5)
  grid.fit(X_train, y_train)
  
  print("Melhores parâmetros:", grid.best_params_)
```
### 8 . Avaliar o modelo
```python
y_pred = grid.predict(X_val)
print("Acurácia:", accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
```
### 9 . Prever no conjunto de teste
``` python
X_test = test[features]
X_test = scaler.transform(X_test)
predictions = grid.predict(X_test)

# Gerar arquivo para submissão
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("Arquivo 'submission.csv' gerado com sucesso!")
```






## Requisitos

- bibliotecas do scikit-learn e pandas, que não vêm por padrão.
  ```python
  pip install pandas numpy scikit-learn
  ```
- Editor de sua preferencia ex.: [Visual Studio Code](https://code.visualstudio.com/download).

## Passos para Configuração

### 1. Baixar e Instalar o Editor

1. Baixe a versão recente do Python a partir do [site oficial](https://www.python.org/).
2. Instale o Editor em sua maquina.
3. Clone esse repositorio na sua pasta com mais privelegios
   ```bash
     git clone https://github.com/UNIVEM-BCC-BSI/titanic.git
   ```

### 2. Iniciar o Projeto

1. Abra o **Editor de Codigo**.
2. Incie o Terminal.

### 3. Executar o Projeto
```bash
  python main.py
```

## Contato

Para dúvidas ou suporte, entre em contato pelo e-mail do desenvolvedor.

