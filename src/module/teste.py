import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt

# Carregar os dados
df = pd.read_csv('../data/College-db.csv', sep=';')

# Remover a coluna "Private" se necessário (caso ela não seja uma variável preditora)
if "Private" in df.columns:
    df = df.drop(columns=["Private"])

if "Dummy" in df.columns:
    df = df.drop(columns=["Dummy"])

# Substituindo as "," por "." na coluna "perc.alumni"
df['perc.alumni'] = df['perc.alumni'].str.replace(",", ".").astype(float)

# Substituindo "Yes", "No" por 1 e 0 na coluna "Apps"
df['Apps'] = df['Apps'].replace('Yes', 1)
df['Apps'] = df['Apps'].replace('No', 0)

# Correlação entre as variáveis
print(df.corr())

# Separar variáveis preditoras (X) e variável resposta (y)
X = sm.add_constant(df.drop(columns=["Apps"]))
y = df['Apps']

model = sm.OLS(y, X)
result = model.fit()

print(result.summary())

plt.scatter(y, result.fittedvalues)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Gráfico de Dispersão: Valores Reais vs. Valores Previstos')
plt.show()