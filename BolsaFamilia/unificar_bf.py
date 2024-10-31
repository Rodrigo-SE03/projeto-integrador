import pandas as pd
import os


# Lista para armazenar todos os DataFrames
df_list = []

# Iterar por cada arquivo CSV na pasta
for arquivo in os.listdir():
    if arquivo.endswith('.csv'):
        # Ler o CSV e adicioná-lo à lista
        df = pd.read_csv(arquivo)
        df_list.append(df)

# Concatenar todos os DataFrames em um único
df_unificado = pd.concat(df_list, ignore_index=True)

# Agrupar por unidade federativa (siglauf) e por anomes
df_agrupado = df_unificado.groupby(['siglauf', 'anomes']).sum().reset_index()

# Salvar o DataFrame final em um arquivo CSV
df_agrupado.to_csv('dados_unificados_por_uf.csv', index=False)

print("Unificação concluída e salva no arquivo 'dados_unificados_por_uf.csv'")
