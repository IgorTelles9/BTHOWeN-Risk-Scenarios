import os
import glob
import pandas as pd

DATA_DIR = "./evaluations/finalizado/"  

# 1) Encontre todos os arquivos CSV que seguem o padrão
csv_pattern = os.path.join(DATA_DIR, "model_evaluation_results_*.csv")
all_files = glob.glob(csv_pattern)

if len(all_files) == 0:
    raise RuntimeError(f"Nenhum arquivo encontrado com o padrão {csv_pattern}")

# 2) Leia todos os CSVs e concatene em um único DataFrame
df_list = []
for file_path in all_files:
    df = pd.read_csv(file_path)
    
    # Opcional: conferir se as colunas essenciais existem em df
    expected_cols = {"portfolio_index", "model_file", "accuracy", "precision", "recall", "f1_score"}
    if not expected_cols.issubset(df.columns):
        raise RuntimeError(f"Arquivo {file_path} não contém todas as colunas esperadas: {expected_cols}")
    
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)

# 3) Agrupe por portfólio e modelo. 
group_cols = ["input_bits", "entry_count", "hash_count", "bpi_count"] 

# 4) Defina quais métricas serão agregadas e quais funções estatísticas usar
agg_dict = {
    "accuracy":   ["mean", "std", "median", "max"],
    "precision":  ["mean", "std", "median", "max"],
    "recall":     ["mean", "std", "median", "max"],
    "f1_score":   ["mean", "std", "median", "max"],
}

# 5) Faça o groupby + agregação
grouped = (
    full_df
    .groupby(group_cols, as_index=False)
    .agg(agg_dict)
)

# 6) Ajuste os nomes das colunas para ficarem "planas" (sem tuplas)
grouped.columns = [
    f"{metric}_{stat}" if metric != "" else stat
    for metric, stat in grouped.columns
]

output_file = "agregado_modelos.csv"
grouped.to_csv(output_file, index=False)
print(f"Arquivo final com {len(grouped)} linhas salvo em: {output_file}")
