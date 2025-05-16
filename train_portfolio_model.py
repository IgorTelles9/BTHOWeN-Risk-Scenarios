#!/usr/bin/env python3

import argparse
import os
import sys

# Garantir que os módulos do BTHOWeN possam ser importados
sys.path.append('./software_model')

from software_model.train_swept_models import create_models

def read_arguments():
    parser = argparse.ArgumentParser(description="Treinar modelos BTHOWeN para previsão de risco no portfólio")
    
    parser.add_argument("--filter_inputs", nargs="+", type=int, default=[8, 16],
                        help="Número de entradas para cada filtro Bloom (aceita múltiplos valores)")
    
    parser.add_argument("--filter_entries", nargs="+", type=int, default=[64, 128, 256],
                        help="Número de entradas em cada filtro Bloom (aceita múltiplos valores; deve ser potência de 2)")
    
    parser.add_argument("--filter_hashes", nargs="+", type=int, default=[3, 5],
                        help="Número de funções hash distintas para cada filtro Bloom (aceita múltiplos valores)")
    
    parser.add_argument("--bits_per_input", nargs="+", type=int, default=[4, 8],
                        help="Número de bits para codificação termométrica para cada entrada (aceita múltiplos valores)")
    
    return parser.parse_args()

def main():
    print("Iniciando treinamento do modelo BTHOWeN para previsão de risco em portfólio")
    
    args = read_arguments()
    
    # Criar diretório models se não existir
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./models/portfolio', exist_ok=True)
    
    for bpi in args.bits_per_input:
        print(f"Executando treinamento com {bpi} bit(s) por entrada")
        create_models(
            'portfolio', args.filter_inputs, args.filter_entries, args.filter_hashes,
            bpi, -1, "risk_model"
        )
    
    print("\nTreinamento concluído! Os modelos estão salvos em ./models/portfolio/")
    print("Para avaliar o modelo, use:")
    print(f"python software_model/evaluate.py ./models/portfolio/{args.save_prefix}_<config>.pickle.lzma portfolio")

if __name__ == "__main__":
    main() 