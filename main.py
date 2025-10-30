import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

label = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train[col] = label.fit_transform(train[col])
    test[col] = label.transform(test[col])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

logreg = LogisticRegression(max_iter=1000)
grid = GridSearchCV(logreg, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)


y_pred = grid.predict(X_val)
print("Acurácia:", accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

X_test = test[features]
X_test = scaler.transform(X_test)
predictions = grid.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("Arquivo 'submission.csv' gerado com sucesso!")
print(output.head())

