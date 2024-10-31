import pandas as pd
import os

# Dicionário de mapeamento de código IBGE para UF (Unidade Federativa)
def get_uf_from_ibge(ibge_code):
    ibge_to_uf = {
        '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA',
        '16': 'AP', '17': 'TO', '21': 'MA', '22': 'PI', '23': 'CE',
        '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE',
        '29': 'BA', '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
        '41': 'PR', '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT',
        '52': 'GO', '53': 'DF'
    }
    return ibge_to_uf.get(str(ibge_code)[:2], 'Desconhecido')

# Lista para armazenar todos os DataFrames
df_list = []

# Iterar por cada arquivo CSV na pasta
for arquivo in os.listdir():
    if arquivo.endswith('.csv'):
        # Ler o CSV
        df = pd.read_csv(arquivo)
        # Adicionar coluna UF com base no código IBGE
        df['UF'] = df['ibge'].apply(get_uf_from_ibge)
        # Adicionar o DataFrame na lista
        df_list.append(df)

# Concatenar todos os DataFrames em um único
df_unificado = pd.concat(df_list, ignore_index=True)

# Agrupar por unidade federativa (UF) e por ano/mês (anomes)
df_agrupado = df_unificado.groupby(['UF', 'anomes']).sum().reset_index()

# Salvar o DataFrame final em um arquivo CSV
df_agrupado.to_csv('dados_unificados_por_uf.csv', index=False)

print("Unificação concluída e salva no arquivo 'dados_unificados_por_uf.csv'")
