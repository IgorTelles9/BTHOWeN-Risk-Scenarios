#!/usr/bin/env python3

import os
import sys
import pickle
import lzma
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
import ctypes as c

sys.path.append('./software_model')

from load_portfolio import prepare_data_for_bthowen

def predict_portfolio_risk(model_fname, portfolio_df=None, portfolio_file=None, bleach=1):
    """
    Faz predições de risco usando um modelo BTHOWeN treinado.
    """
    print("Carregando modelo...")
    with lzma.open(model_fname, "rb") as f:
        state_dict = pickle.load(f)
    
    model = state_dict["model"]
    model_info = state_dict["info"]
    
    # Configurar o bleaching
    model.set_bleaching(bleach)
    
    # Verificar se o portfólio foi fornecido ou se precisa ser carregado
    if portfolio_df is None and portfolio_file is not None:
        print(f"Carregando portfólio de {portfolio_file}...")
        portfolio_df = pd.read_parquet(portfolio_file)
    elif portfolio_df is None and portfolio_file is None:
        print("Carregando portfólio padrão...")
        try:
            portfolio_path = '../portfolio.parquet'
            if not os.path.exists(portfolio_path):
                portfolio_path = 'portfolio.parquet'
            portfolio_df = pd.read_parquet(portfolio_path)
        except Exception as e:
            print(f"Erro ao carregar portfólio: {e}")
            return None
    
    print("Preparando dados para predição...")
    
    df = portfolio_df.copy()
    original_index = df.index
    
    # Salvar a coluna 'risk' atual para comparação posterior
    if 'risk' in df.columns:
        true_risk = df['risk'].copy()
    else:
        true_risk = None
    
    # Normalizar features
    feature_cols = [col for col in df.columns if col != 'risk' and col != 'Date']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Extrair features
    features = df.drop(columns=['risk'] if 'risk' in df.columns else []).values
    
    # Binarizar as features usando codificação termométrica
    bits_per_input = model_info["bits_per_input"]
    std_skews = [norm.ppf((i+1)/(bits_per_input+1)) for i in range(bits_per_input)]
    
    mean_features = np.mean(features, axis=0)
    std_features = np.std(features, axis=0)
    
    binary_features = []
    for i in std_skews:
        binary_features.append((features >= mean_features+(i*std_features)).astype(c.c_ubyte))
    
    # Concatenar as features binarizadas
    binary_features = np.concatenate(binary_features, axis=1)
    
    # Fazer predições
    print("Realizando predições...")
    predictions = []
    confidence = []
    
    for i in range(len(binary_features)):
        # Pegar as respostas de cada discriminador (0=não risco, 1=risco)
        responses = [model.discriminators[j].predict(binary_features[i]) for j in range(len(model.discriminators))]
        
        # Calcular confiança (diferença entre respostas)
        conf = abs(responses[1] - responses[0]) / (responses[0] + responses[1] + 1e-10)
        confidence.append(conf)
        
        # Pegar a classe predita (discriminador com maior resposta)
        pred_classes = model.predict(binary_features[i])
        predictions.append(pred_classes[0])
    
    # Criar DataFrame de resultados
    result_df = pd.DataFrame({
        'predicted_risk': predictions,
        'confidence': confidence
    }, index=original_index)
    
    # Adicionar comparação com valores reais, se disponíveis
    if true_risk is not None:
        result_df['true_risk'] = true_risk
        result_df['correct'] = result_df['predicted_risk'] == result_df['true_risk']
        
        # Calcular acurácia
        accuracy = result_df['correct'].mean() * 100
        print(f"Acurácia do modelo: {accuracy:.2f}%")
        
        # Calcular precisão e recall para cenários de risco (classe 1)
        true_positives = ((result_df['predicted_risk'] == 1) & (result_df['true_risk'] == 1)).sum()
        false_positives = ((result_df['predicted_risk'] == 1) & (result_df['true_risk'] == 0)).sum()
        false_negatives = ((result_df['predicted_risk'] == 0) & (result_df['true_risk'] == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"Precisão para cenários de risco: {precision:.2f}")
        print(f"Recall para cenários de risco: {recall:.2f}")
    
    return result_df

def read_arguments():
    parser = argparse.ArgumentParser(description="Prever risco em portfólio usando modelo BTHOWeN treinado")
    
    parser.add_argument("model_path", help="Caminho para o arquivo do modelo treinado (.pickle.lzma)")
    
    parser.add_argument("--portfolio", default=None, help="Caminho para arquivo parquet com dados de portfólio (opcional)")
    
    parser.add_argument("--bleach", type=int, default=1, help="Valor de bleaching para o modelo (padrão: 1)")
    
    parser.add_argument("--output", default=None, help="Caminho para salvar as predições (opcional)")
    
    return parser.parse_args()
def main():
    # args = read_arguments()
    
    result = predict_portfolio_risk(
        model_fname='./models/portfolio/portfolio5.pickle.lzma',
        portfolio_file='./portfolio.parquet',
        bleach=1
    )
    
    if result is not None:
        # Exibir primeiras linhas do resultado
        print("\nPrimeiras linhas das predições:")
        print(result.head())
        
        # Salvar resultados, se solicitado
        result.to_csv('predict.csv')
        print(f"Resultados salvos em predict.csv")

if __name__ == "__main__":
    main() 