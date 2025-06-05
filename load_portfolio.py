import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Preparar dados para BTHOWeN
def prepare_data_for_bthowen(portfolio_df):
    """
    Prepara os dados do portfólio para uso com o modelo BTHOWeN.
    """
    df = portfolio_df.copy()
    
    # Normalização de features 
    feature_cols = [col for col in df.columns if col != 'risk' and col != 'Date']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Remover a coluna de risco (target) e converter para array numpy
    features = df.drop(columns=['risk']).values
    # Converter risco (boolean) para inteiro (0=False, 1=True)
    labels = df['risk'].astype(int).values
    
    # Criar lista de tuplas (features, label)
    dataset = [(features[i], labels[i]) for i in range(len(features))]
    
    # Dividir em conjuntos de treino e teste de forma aleatória
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    print(f"Dataset preparado: {len(train_dataset)} amostras de treino, {len(test_dataset)} amostras de teste")
    print(f"Dimensão das features: {features.shape[1]}")
    print(f"Distribuição das classes: {np.sum([l for _, l in dataset])} positivos, {len(dataset) - np.sum([l for _, l in dataset])} negativos")
    
    return train_dataset, test_dataset

# Carregar o dataset do portfólio
try:
    # Tentamos primeiro com o caminho relativo
    portfolio_path = '../portfolio.parquet'
    if not os.path.exists(portfolio_path):
        # Se não encontrar, tentamos com caminho absoluto
        portfolio_path = 'portfolio.parquet'
    
    portfolio = pd.read_parquet(portfolio_path)
    
    # Se 'Date' estiver no índice, reseta para coluna
    if portfolio.index.name == 'Date':
        portfolio = portfolio.reset_index()
    
    # Remove a coluna de data se existir para preparação para BTHOWeN
    if 'Date' in portfolio.columns:
        portfolio = portfolio.drop(columns=['Date'])
    
    print(f"Portfólio carregado com sucesso: {len(portfolio)} registros, {portfolio.columns.size} colunas")
    
except Exception as e:
    print(f"Erro ao carregar arquivo de portfólio: {e}")
    print("Certifique-se de que o arquivo portfolio.parquet existe no diretório apropriado")

# Para testes do módulo
if __name__ == "__main__":
    try:
        train_data, test_data = prepare_data_for_bthowen(portfolio)
        print(f"Formato dos dados de treino: {len(train_data)} amostras")
        print(f"Exemplo de amostra de treino - shape: {train_data[0][0].shape}, label: {train_data[0][1]}")
    except Exception as e:
        print(f"Erro ao preparar dados: {e}")

