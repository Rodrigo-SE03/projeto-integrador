import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

umidade = 'umidade.csv'
incendio = 'incendio.csv'

incendio = pd.read_csv(incendio)
umidade = pd.read_csv(umidade)
dt = pd.merge(umidade, incendio, on='data_pas', how='left').fillna(0)
#new column, has_fire, 1 if has fire, 0 if not
dt['has_fire'] = dt['focos'].apply(lambda x: 1 if x > 0 else 0)

dt.drop(columns=['TEMPERATURA DO PONTO DE ORVALHO (°C)', 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)', 'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
                 'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)', 'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)', 
                 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)', 'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)', 'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)', 
                 'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)', 'data_pas'], inplace=True)
dt.head()
dt.columns = ['precipitação', 'pressão atmosferica', 'temperatura do ar', 'umidade relativa', 'vento, direcao', 'vento, rajada', 'vento, velocidade', 'focos', 'has_fire']
#reorder columns
dt = dt[['has_fire', 'focos', 'temperatura do ar', 'umidade relativa', 'vento, direcao', 'vento, rajada', 'vento, velocidade', 'precipitação', 'pressão atmosferica']]
dt.to_csv('dataset.csv', index=False)