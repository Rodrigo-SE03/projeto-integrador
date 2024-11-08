import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from linear_regression import LinearRegression
from read_data import get_data


#Listas de chaves e constantes
keys_csv = {
    "anomes": "Ano/Mês",
    "qtd_ben_bas": "Quantidade de benefícios básicos",
    "qtd_ben_var": "Quantidade de benefícios variáveis",
    "qtd_ben_bvj": "Quantidade de benefícios jovem",
    "qtd_ben_bvn": "Quantidade de benefícios nutriz",
    "qtd_ben_bvg": "Quantidade de benefícios gestantes",
    "qtd_ben_bsp": "Quantidade de benefícios superação extrema pobreza"
}

estados = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]

regioes_geograficas = {
    'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
    'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
    'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
    'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
    'Sul': ['PR', 'RS', 'SC']
}

regioes_socioeconomicas = {
    'Amazônia': ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO', 'MT', 'MA'],
    'Centro-Sul': ['DF', 'GO', 'MS', 'MT', 'PR', 'RS', 'SC', 'SP', 'RJ', 'ES', 'MG'],
    'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE']
}


bolsa_keys = ['qtd_ben_bas', 'qtd_ben_var', 'qtd_ben_bvj', 'qtd_ben_bvn', 'qtd_ben_bvg', 'qtd_ben_bsp']


# Configuração da página
st.set_page_config(
    page_title="Projeto Integrador",
)

st.title("Análise de Regressão Linear Múltipla do CadÚnico e Bolsa Família")
st.write("Esta aplicação permite realizar e visualizar uma análise de regressão linear múltipla entre os dados do CadÚnico e do Bolsa Família.")


#Filtros para análise
analises = ["Geral", "Por Região Geográfica", "Por Região Socioeconômica", "Por Estado"]
tipo_analise = st.sidebar.selectbox("Escopo de Análise", analises, placeholder="Selecione um escopo de análise")

keys_to_analyze = []
if tipo_analise == "Geral":
    keys_to_analyze = estados

elif tipo_analise == "Por Região Geográfica":
    regiao = st.sidebar.selectbox("Região Geográfica", regioes_geograficas.keys(), placeholder="Selecione uma região")
    keys_to_analyze = regioes_geograficas[regiao]

elif tipo_analise == "Por Região Socioeconômica":
    regiao = st.sidebar.selectbox("Região Socioeconômica", regioes_socioeconomicas.keys(), placeholder="Selecione uma região")
    keys_to_analyze = regioes_socioeconomicas[regiao]

else:
    estado = st.sidebar.selectbox("Estado", estados, placeholder="Selecione uma região")
    keys_to_analyze = [estado]

nivel = st.sidebar.selectbox("Nivel de analise", ['Pessoa', 'Família'], placeholder="Selecione o nivel de analise")

variavel_dependente = st.sidebar.selectbox("Variável Dependente", ['Total CadUnico', 'Total até meio salário mínimo', 'Total na linha da pobreza', 'Total extremamente pobre', 'Total pobreza e extrema pobreza'], placeholder="Selecione uma variável independente")
variavel_independente = st.sidebar.selectbox("Variável Independente", bolsa_keys, placeholder="Selecione uma variável independente")

keys_csv_check = [variavel_independente]
# for idx, k in enumerate(bolsa_keys):
#     keys_csv_check.append(st.sidebar.checkbox(keys_csv[k], value=True, key=k))


# Carregar os dados
#b_to_write = [x for i, x in enumerate(bolsa_keys) if keys_csv_check[i]]
b_to_write = [variavel_independente]
cad, bolsa  = get_data(keys_to_analyze, nivel, [variavel_dependente], b_to_write)

X = cad[variavel_dependente].values
y = bolsa[b_to_write].values

#X, y = datasets.make_regression(n_samples=100, n_features=5, noise=10, random_state=4)


# Realizar a regressão
model = LinearRegression(X, y)



# Exibir o resumo dos resultados
st.subheader("Resultados da Regressão")
st.write(f"R²: {model.r2:.4f}")
st.write(f"R² Ajustado: {model.adj_r2:.4f}")

#variaveis = ["Intercepto"] + [keys_csv[k] for i, k in enumerate(bolsa_keys) if keys_csv_check[i]]
variaveis = ["Intercepto"] + [keys_csv[variavel_independente]]
table = pd.DataFrame(model.b, columns=["Coeficiente"], index=variaveis)
# table["P-Value"] = model.p
st.table(table)

#Se 2d, exibir regressão
if len(b_to_write) == 1:
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue")
    ax.plot(X, model.y_pred, color="red")
    ax.set_ylabel(keys_csv[b_to_write[0]])
    ax.set_xlabel(variavel_dependente)
    ax.set_title("Regressão Linear")
    st.pyplot(fig)

    
# Análise dos resíduos
st.subheader("Análise dos Resíduos")
residuals = model.residuals
fig, ax = plt.subplots()
ax.hist(residuals, bins=20, edgecolor="black")
ax.set_title("Distribuição dos Resíduos")
ax.set_xlabel("Resíduos")
ax.set_ylabel("Frequência")
st.pyplot(fig)

# Gráfico de dispersão dos resíduos vs valores ajustados
st.subheader("Resíduos vs Valores Ajustados")
fitted_values = model.y_pred
fig, ax = plt.subplots()
ax.scatter(fitted_values, residuals)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Valores Ajustados")
ax.set_ylabel("Resíduos")
ax.set_title("Resíduos vs Valores Ajustados")
st.pyplot(fig)