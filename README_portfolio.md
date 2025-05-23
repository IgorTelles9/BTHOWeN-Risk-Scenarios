# Previsão de Risco em Portfólio com BTHOWeN

Este guia explica como usar o modelo BTHOWeN (Bloom-filter-based THermOmeter-encoded Weightless Neural Network) para prever cenários de risco em um portfólio de investimentos.

## Visão Geral

O BTHOWeN é um modelo de rede neural sem pesos que utiliza filtros de Bloom e codificação termométrica para classificação. Nesta aplicação, o adaptamos para prever cenários de risco em um portfólio de investimentos.

## Estrutura de Arquivos

- `load_portfolio.py`: Carrega e prepara os dados do portfólio para o BTHOWeN
- `train_portfolio_model.py`: Script para treinar múltiplos modelos BTHOWeN
- `predict_risk.py`: Script para fazer predições usando um modelo treinado
- `software_model/`: Implementação do modelo BTHOWeN
- `tabular_tools.py`: Modificado para suportar o dataset do portfólio

## Uso

### Treinamento do Modelo

Para treinar modelos BTHOWeN com diferentes configurações:

```bash
python train_portfolio_model.py [opções]
```

Opções disponíveis:
- `--filter_inputs`: Número de entradas para cada filtro Bloom (padrão: 8 16)
- `--filter_entries`: Número de entradas em cada filtro (padrão: 64 128 256)
- `--filter_hashes`: Número de funções hash para cada filtro (padrão: 3 5)
- `--bits_per_input`: Número de bits para codificação termométrica (padrão: 4 8)
- `--save_prefix`: Prefixo para salvar os modelos (padrão: risk_model)
- `--num_workers`: Número de processos em paralelo (padrão: -1, usa todos os CPUs)

Exemplo:
```bash
python train_portfolio_model.py --filter_inputs 8 16 --filter_entries 128 256 --filter_hashes 3 5 --bits_per_input 4 8
```

### Realizando Predições

Para fazer predições com um modelo treinado:

```bash
python predict_risk.py [modelo_treinado] [opções]
```

Opções disponíveis:
- `model_path`: Caminho para o arquivo do modelo treinado (obrigatório)
- `--portfolio`: Caminho para arquivo parquet com dados (opcional)
- `--bleach`: Valor de bleaching para o modelo (padrão: 1)
- `--output`: Caminho para salvar as predições (opcional)

Exemplo:
```bash
python predict_risk.py ./models/portfolio/risk_model_8input_128entry_3hash_4bpi.pickle.lzma --output predictions.csv
```

## Explicação dos Parâmetros

O desempenho do modelo BTHOWeN depende de vários parâmetros:

- **bits_per_input**: Controla a resolução da codificação termométrica. Valores maiores capturam mais variações nos dados.
- **filter_inputs**: Número de entradas booleanas para cada LUT/filtro no modelo.
- **filter_entries**: Tamanho dos arrays de armazenamento para os filtros. Valores maiores reduzem falsos positivos.
- **filter_hashes**: Número de funções hash para cada filtro. Afeta a precisão e o desempenho.
- **bleach**: Valor de limite para membros que não foram potencialmente vistos pelo menos b vezes.

## Avaliação do Modelo

O script `predict_risk.py` calcula e exibe:
- Acurácia geral do modelo
- Precisão para cenários de risco (quão confiáveis são os alertas de risco)
- Recall para cenários de risco (quão bem o modelo identifica todos os cenários de risco reais)

## Interpretação dos Resultados

O modelo produz duas principais saídas:
- `predicted_risk`: Classificação binária (0 = sem risco, 1 = risco)
- `confidence`: Confiança da predição (diferença relativa entre as respostas dos discriminadores)

Alertas de risco com alta confiança devem ser priorizados para ações de mitigação de risco no portfólio.

## Limitações e Considerações

- O modelo BTHOWeN tem um comportamento determinístico, o que pode limitar sua capacidade de generalização.
- A qualidade das predições depende fortemente da escolha adequada dos hiperparâmetros.
- Dados históricos podem não refletir condições futuras do mercado.