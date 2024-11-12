import pandas as pd

umidade = 'umidade.csv'
incendio = 'incendio.csv'

incendio = pd.read_csv(incendio)
umidade = pd.read_csv(umidade)

dt = pd.merge(incendio, umidade, on='data_pas', how='left')
dt = dt.dropna()

dt['data_pas'] = pd.to_datetime(dt['data_pas'])
dt['data_pas'] = dt['data_pas'].dt.hour
dt = dt.rename(columns={'data_pas': 'HORÁRIO'})

def get_data(keys:list):
    x = dt[keys]
    y = dt['focos']
    return x, y

def get_correlation(keys:list):
    return dt[keys].corr()

def get_dataset(keys:list, pred):
    keys.append('focos')
    dt_tmp = dt[keys].copy()
  
    #add new columns pred to the dataset
    dt_tmp['pred'] = pred
    return dt_tmp