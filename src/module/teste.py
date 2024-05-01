import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carregar os dados
df = pd.read_csv('../data/College-db.csv', sep=';')

# Remover a coluna "Private" se necessário (caso ela não seja uma variável preditora)
if "Private" in df.columns:
    df = df.drop(columns=["Private"])

# Separar variáveis preditoras (X) e variável resposta (y)
X = df.drop(columns=["Apps"])
print(X)
y = df['Apps'].map({'Yes': 1, 'No': 0})

# Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões com o conjunto de teste
y_pred = model.predict(X_test)

# Calcular o erro quadrático médio (MSE) para avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio (MSE):", mse)
print(df.describe())