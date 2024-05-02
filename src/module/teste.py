import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt

# Carregar os dados
df = pd.read_csv('../data/pinguins.csv', sep=';')

df = df.drop(columns=["especie", "ilha", "comprimento_bico", "sexo", "ano"])
df['profundidade_bico'] = df['profundidade_bico'].str.replace(",", ".").astype(float)
df = df.dropna()

print(df.info())
print(df.head())
print(df.corr())

X = sm.add_constant(df.drop(columns=["comprimento_nadadeira"]))
y = df["comprimento_nadadeira"]

model = sm.OLS(y, X)
result = model.fit()

print(result.summary())

plt.scatter(y, result.fittedvalues)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Gráfico de Dispersão: Valores Reais vs. Valores Previstos')
plt.show()