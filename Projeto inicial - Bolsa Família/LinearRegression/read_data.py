import pandas as pd
import matplotlib.pyplot as plt

cad_unico_path = 'data/cadUnico.csv'
bolsa_familia_path = 'data/BolsaFamilia.csv'

with open(cad_unico_path, 'r') as f:
    cad_unico = pd.read_csv(f)

with open(bolsa_familia_path, 'r') as f:
    bolsa_familia = pd.read_csv(f)


def get_data(ufs, nivel, keys_cad, keys_bolsa):

    data_cad = cad_unico[cad_unico['UF'].isin(ufs)]
    
    if nivel == 'Pessoa': 
        data_cad = data_cad[['anomes', 'cadunico_tot_pes', 'cadunico_tot_pes_rpc_ate_meio_sm', 'cadunico_tot_pes_pob', 'cadunico_tot_pes_ext_pob', 'cadunico_tot_pes_pob_e_ext_pob']]
    else:
        data_cad = data_cad[['anomes', 'cadunico_tot_fam', 'cadunico_tot_fam_rpc_ate_meio_sm', 'cadunico_tot_fam_pob', 'cadunico_tot_fam_ext_pob', 'cadunico_tot_fam_pob_e_ext_pob']]
    
    data_cad = data_cad.groupby('anomes').sum().reset_index()
    data_cad = data_cad.dropna()
    data_cad.columns = ['AnoMes', 'Total CadUnico', 'Total até meio salário mínimo', 'Total na linha da pobreza', 'Total extremamente pobre', 'Total pobreza e extrema pobreza']
    data_cad = data_cad.drop(columns=[col for col in data_cad.columns if col not in keys_cad])


    data_bolsa = bolsa_familia[bolsa_familia['siglauf'].isin(ufs)]
    data_bolsa = data_bolsa.groupby('anomes').sum().reset_index()
    data_bolsa = data_bolsa.drop(columns=[col for col in data_bolsa.columns if col not in keys_bolsa])
    return data_cad, data_bolsa
