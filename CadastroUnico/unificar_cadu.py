import pandas as pd
import os

# Função para mapear o código IBGE para UF
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

# Lista para armazenar os DataFrames de cada arquivo
df_list = []

# Iterar sobre todos os arquivos CSV na pasta
for arquivo in os.listdir():
    if arquivo.endswith('.csv'):
        # Ler o arquivo CSV
        df = pd.read_csv(arquivo)
        # Adicionar coluna UF com base no código IBGE
        df['UF'] = df['ibge'].apply(get_uf_from_ibge)
        # Remover a coluna "ibge" pois ela não é necessária no resultado final
        df = df.drop(columns=['ibge'])
        # Adicionar o DataFrame à lista
        df_list.append(df)

# Concatenar todos os DataFrames em um único DataFrame
df_unificado = pd.concat(df_list, ignore_index=True)
df_condensado = df_unificado.groupby(['UF', 'anomes'], as_index=True).sum().reset_index()

# Salvar o DataFrame final em um novo arquivo CSV
df_condensado.to_csv('results/dados_agrupados_por_uf_e_anomes.csv', index=False)

print("Arquivo final criado e salvo como 'dados_agrupados_por_uf_e_anomes.csv'")
