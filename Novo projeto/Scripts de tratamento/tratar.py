import pandas as pd
import os

def clean_and_split_date(file_path):
    # Carregar o arquivo CSV com encoding apropriado e delimitador
    data = pd.read_csv(file_path, delimiter=';', encoding='latin1', skiprows=3)
    
    # Remover as últimas 10 linhas
    data = data[:-8]

    # Renomear a coluna para um nome consistente
    data.rename(columns={data.columns[0]: 'Ano/mês atendimento'}, inplace=True)

    # Detectar o ano da primeira linha com dados
    first_year = data.iloc[0]['Ano/mês atendimento']
    if first_year.isdigit():
        first_year = int(first_year)
    else:
        first_year = int(first_year.split('/')[1])  # Caso a célula tenha mais informações

    # Remover linhas que contêm apenas o ano ou que pertencem ao ano inicial
    data = data[~data['Ano/mês atendimento'].astype(str).str.match(r'^\d{4}$')]
    data = data[~data['Ano/mês atendimento'].str.contains(str(first_year))]

    # Separar a coluna "Ano/mês atendimento" em "Ano" e "Mês" usando uma função de extração condicional
    data['Ano'] = data['Ano/mês atendimento'].str.extract(r'(\d{4})$')[0].astype(int)  # Extrai o ano
    data['Mês'] = data['Ano/mês atendimento'].str.extract(r'^(.*?)/')[0]  # Extrai o mês, se presente
    
    # Limpar o nome do mês (remover ".." ou caracteres indesejados)
    data['Mês'] = data['Mês'].str.replace(r'\.\.', '', regex=True).str.strip()

    # Identificar anos com valores faltantes ("-")
    anos_para_remover = data.loc[data.isin(["-"]).any(axis=1), 'Ano'].unique()

    # Remover linhas que pertencem aos anos identificados
    data = data[~data['Ano'].isin(anos_para_remover)]
    
    # Remover a coluna original "Ano/mês atendimento"
    data.drop(columns=['Ano/mês atendimento'], inplace=True)

    # Reordenar as colunas para ter "Ano" e "Mês" como as primeiras colunas
    cols = ['Ano', 'Mês'] + [col for col in data.columns if col not in ['Ano', 'Mês']]
    data = data[cols]

    # Retornar o DataFrame limpo e com as colunas separadas
    return data

def consolidate_files(folder_path, output_file):
    # Lista para armazenar os DataFrames de cada arquivo
    all_data = []

    # Percorrer todos os arquivos CSV na pasta especificada
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            # Limpar e processar o arquivo
            cleaned_data = clean_and_split_date(file_path)
            # Adicionar ao consolidado
            all_data.append(cleaned_data)
    
    # Concatenar todos os DataFrames
    consolidated_data = pd.concat(all_data, ignore_index=True)
    
    # Salvar o arquivo final consolidado
    consolidated_data.to_csv(output_file, index=False)

# # Exemplo de uso do script para um único arquivo
# file_path = '../Dados brutos/2015-2013.csv'
# cleaned_data = clean_and_split_date(file_path)
# cleaned_data.to_csv('../Dados tratados/cleaned_data.csv', index=False)

# # Exemplo de uso do script para toda a pasta
folder_path = '../Dados brutos/'  # Caminho para a pasta com os arquivos
output_file = '../Dados tratados/condensado.csv'  # Nome do arquivo consolidado
consolidate_files(folder_path, output_file)