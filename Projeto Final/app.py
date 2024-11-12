import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from linear_regression import LinearRegression
import seaborn as sns
from read_data import get_data, get_correlation, get_dataset
import numpy as np

def set_config(page_title, title, description):
    st.set_page_config(page_title=page_title, layout="wide")
    st.title(title)
    st.write(description)


def load_data(mes, uf):
    x, y = get_data(mes, uf)
    return x, y


def plot_data(x, y, pred, x_name='X', y_name='Y'):
    fig, ax = plt.subplots()
    ax.grid(True)

    bg = st.get_option("theme.backgroundColor")
    text = st.get_option("theme.textColor")

    if bg and text:
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.spines['bottom'].set_color(text)
        ax.spines['top'].set_color(text)
        ax.spines['right'].set_color(text)
        ax.spines['left'].set_color(text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)
        ax.tick_params(axis='x', colors=text)
        ax.tick_params(axis='y', colors=text)
        ax.title.set_color(text)
        ax.title.set_fontsize(20)
        ax.title.set_fontweight('bold')

    ax.scatter(x, y, color='green')
    ax.plot(x,pred, color='yellow')

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    st.pyplot(fig)


def show_results(model:LinearRegression, variavies_names:list, y_name = 'Y'):
    st.subheader("Resultados da Regressão")
    st.write(f"R²: {model.r2:.4f}")

    table = pd.DataFrame(model.b, columns=["Coeficiente"], index=variavies_names)
    table['P-Valor'] = model.p
    st.table(table)

    if len(model.b) == 2:
        plot_data(model.x[:, 1], model.y, model.y_pred, variavies_names[1], y_name)
        
    else:
        #graph of predictec X real
        fig, ax = plt.subplots()
        ax.grid(True)
        
        bg = st.get_option("theme.backgroundColor")
        text = st.get_option("theme.textColor")
        
        if bg and text:
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.spines['bottom'].set_color(text)
            ax.spines['top'].set_color(text)
            ax.spines['right'].set_color(text)
            ax.spines['left'].set_color(text)
            ax.xaxis.label.set_color(text)
            ax.yaxis.label.set_color(text)
            ax.title.set_color(text)
            ax.tick_params(axis='x', colors=text)
            ax.tick_params(axis='y', colors=text)
            ax.title.set_color(text)
            ax.title.set_fontsize(20)
            ax.title.set_fontweight('bold')

        ax.scatter(model.y, model.y_pred, color='green')
        ax.plot([model.y.min(), model.y.max()], [model.y.min(), model.y.max()], color='yellow')
        ax.set_xlabel('Real')
        ax.set_ylabel('Predito')
        st.pyplot(fig)


if __name__ == '__main__':
    
    set_config(page_title="Projeto Integrador", 
        title="Análise de Regressão Linear Sobre Condições Climáticas e Incêndios Florestais", 
        description="Esta aplicação permite a análise de regressão linear sobre dados de incêndios florestais e condições climáticas.")
    
    keys = ['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
            'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
            'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
            'TEMPERATURA DO PONTO DE ORVALHO (°C)',
            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
            'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
            'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)',
            'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
            'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
            'UMIDADE RELATIVA DO AR, HORARIA (%)',
            'VENTO, DIREÇÃO HORARIA (gr) (° (gr))', 'VENTO, RAJADA MAXIMA (m/s)',
            'VENTO, VELOCIDADE HORARIA (m/s)']

    st.sidebar.title("Selecione os Dados")
    keys_selection = st.sidebar.multiselect('Selecione os Dados', keys, default=['UMIDADE RELATIVA DO AR, HORARIA (%)'], placeholder='Selecione os dados')

    #Checkbox to see the correlation matrix
    if len(keys_selection) > 1:
        if st.sidebar.checkbox("Verificar Correlação"):
            corr = get_correlation(keys_selection)
            fig = plt.figure(figsize = (10,8))

            #configuring the colors
            bg = st.get_option("theme.backgroundColor")
            text = st.get_option("theme.textColor")

            if bg and text:
                fig.patch.set_facecolor(bg)
                ax = fig.add_subplot(111)
                ax.set_facecolor(bg)
                ax.spines['bottom'].set_color(text)
                ax.spines['top'].set_color(text)
                ax.spines['right'].set_color(text)
                ax.spines['left'].set_color(text)
                ax.xaxis.label.set_color(text)
                ax.yaxis.label.set_color(text)
                ax.title.set_color(text)
                ax.tick_params(axis='x', colors=text)
                ax.tick_params(axis='y', colors=text)
                ax.title.set_color(text)
                ax.title.set_fontsize(20)
                ax.title.set_fontweight('bold')

            sns.heatmap(corr, annot=True, cmap='coolwarm')
            st.pyplot(fig)



    if len(keys_selection) == 0:
        st.warning("Selecione ao menos um dado")
    
    else:
        x, y  = get_data(keys_selection)
        model = LinearRegression(x, y)
        show_results(model, ['Intercepto'] + keys_selection, 'Focos de Incêndio')