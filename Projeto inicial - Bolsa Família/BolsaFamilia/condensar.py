import pandas as pd

# Carregar o arquivo condensado
df = pd.read_csv('results/dados_unificados_por_uf_bf.csv')

# Remover a coluna de UF, pois queremos apenas os totais por ano
df = df.drop(columns=['siglauf'])

# Agrupar apenas pela coluna "ano" para somar os dados de todas as UFs
df_total_por_ano = df.groupby('anomes').sum().reset_index()

# Salvar o DataFrame consolidado em um novo arquivo CSV
df_total_por_ano.to_csv('results/dados_totais_por_mes_bf.csv', index=False)

print("Dados consolidados e salvos no arquivo 'dados_totais_por_mes_bf.csv'")
