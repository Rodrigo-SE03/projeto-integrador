import pandas as pd
import streamlit as st
from map import render_map
from openweather import OpenWeather



def load_model():
    
    tipo_modelo = st.selectbox("Modelo", ["PyTorch", "Scikit-learn"])
    if 'is_personalized' not in st.session_state or st.session_state['is_personalized'] != tipo_modelo:
        st.session_state['model'] = None
        st.session_state['is_personalized'] = tipo_modelo

    if "model" not in st.session_state or st.session_state['model'] is None:
        from regressao_logistica import get_model
        model = get_model(tipo_modelo == "PyTorch")
        st.session_state['model'] = model
    else: model = st.session_state['model']

    return model


def sidebar():
    paths = {
        "incendios": "dados/incendio.csv",
        "umidade": "dados/umidade.csv",
        "merged" : "dados/dataset.csv"
    }

    if "df_incendios" not in st.session_state:
        st.session_state["df_incendios"] = pd.read_csv(paths["incendios"])
    if "df_umidade" not in st.session_state:
        st.session_state["df_umidade"] = pd.read_csv(paths["umidade"])
    if "df_merged" not in st.session_state:
        st.session_state["df_merged"] = pd.read_csv(paths["merged"])


    st.sidebar.write("# Datasets")
    with st.sidebar.expander("Dataset de incêndios"):
        st.write(st.session_state["df_incendios"])

    with st.sidebar.expander("Dataset do tempo"):
        st.write(st.session_state["df_umidade"])

    with st.sidebar.expander("Dataset unificado"):
        st.write(st.session_state["df_merged"])


st.set_page_config(
    page_title="Regressão Logística",
    page_icon=":bar_chart:",
    initial_sidebar_state="collapsed"
)


st.title("Regressão Logística")
st.markdown("Modelo de regressão logística para avaliar a probabilidade de se haver focos de incendio dada a temperatura e a umidade relativa do ar.")

#Expanded to explain what is a Regressão logística
with st.expander("O que é uma regressão logística?"):
    st.markdown("A regressão logística é um modelo de regressão utilizado para prever a probabilidade de uma variável dependente categórica. Em problemas de classificação binária, a regressão logística modela a probabilidade de a variável dependente pertencer a uma classe específica, como por exemplo, a probabilidade de um e-mail ser spam ou não spam.")

    #Descrever como é calculada a probabilidade
    st.markdown("A probabilidade de um evento ocorrer é calculada pela função logística, que é a função sigmóide:")
    #função mutipla
    st.latex(r'f(x) = \frac{1}{1 + e^{\beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n}}')
    
    st.markdown("Onde:")

    st.latex(r'f(x) = \text{Probabilidade do evento ocorrer}')
    st.latex(r'\beta_0 = \text{Intercepto}')
    st.latex(r'\beta_1, \beta_2, \ldots, \beta_n = \text{Coeficientes}')
    st.latex(r'x_1, x_2, \ldots, x_n = \text{Variáveis independentes}')

    st.markdown("A função logística é uma função sigmóide que retorna valores entre 0 e 1, que pode ser chamado de razão de chance. Se a probabilidade calculada for maior que um limiar (threshold), o evento é classificado como positivo, caso contrário, é classificado como negativo.")


model = load_model()
threshold = st.number_input("Threshold", min_value=0.00, max_value=1.00, value=0.520, step=0.001 ,format="%.3f")

metricas = model.metrics(threshold=threshold)
st.write("## Métricas")


with st.expander("Significado das métricas"):
    st.markdown("As métricas são calculadas a partir da matriz de confusão, que é uma tabela que mostra as frequências de classificação para cada classe do modelo.")
    st.markdown("As métricas calculadas são:")


    st.markdown("- **Acurácia**: Proporção de previsões corretas.")

    st.latex(r'Accuracy = \frac{TP + TN}{TP + TN + FP + FN}')

    st.markdown("- **Precisão**: Proporção de previsões corretas de focos de incêndio.")

    st.latex(r'Precision = \frac{TP}{TP + FP}')

    st.markdown("- **Recall**: Proporção de focos de incêndio corretamente previstos.")

    st.latex(r'Recall = \frac{TP}{TP + FN}')

    st.markdown("Onde:")
    st.markdown("- **Verdadeiro Positivo (TP)**: Número de previsões corretas de focos de incêndio.")
    st.markdown("- **Falso Positivo (FP)**: Número de previsões incorretas de focos de incêndio.")
    st.markdown("- **Verdadeiro Negativo (TN)**: Número de previsões corretas de não focos de incêndio.")
    st.markdown("- **Falso Negativo (FN)**: Número de previsões incorretas de não focos de incêndio.")


st.write("Acurácia: ", str(round(metricas["accuracy"], 3)*100) + "%")
st.write("Precisão: ", str(round(metricas["precision"], 3)*100) + "%")
st.write("Recall: ", str(round(metricas["recall"], 3)*100) + "%")
sidebar()

st.write("## Mapa de Goiás")
st.write("Clique no mapa para realizar a previsão de incêndio em uma determinada localidade.")

coords = render_map(6)

openweather = OpenWeather()
if coords:
    lat, lon = coords
    if coords == [0,0] or lat is None or lon is None: st.stop()
    forecast = openweather.forecast(lat, lon)

    st.write("## Probabilidade de incêndio")
    propabilidades = []

    table = pd.DataFrame(columns=["Data", "Haverá focos de incêndio", " Razão de Chance", "Threshold"])

    for prevision in forecast:
        line = []
        prob = model.predict(prevision[1:])
        propabilidades.append(prob)
        line.append(prevision[0])

        has_focos = prob[0] == 1
        line.append("Sim" if has_focos else "Não")
        line.append(f'{prob[1][0]:.4f}')
        line.append(threshold)

        table.loc[len(table)] = line
    
    st.write(table)

