import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------ configuração ------------------------

HYPERPARAMS = ["input_bits_", "entry_count_", "hash_count_", "bpi_count_"]   
METRICS      = ["accuracy_mean", "precision_mean", "recall_mean", "f1_score_mean"]

# ------------------------ funções auxiliares ------------------

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Remove % e transforma em float
    if df["accuracy_mean"].dtype == object:
        df["accuracy_mean"] = df["accuracy_mean"].str.replace("%", "").astype(float)

    # Garante tipo numérico onde faz sentido
    for col in HYPERPARAMS + METRICS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna média e desvio-padrão das métricas"""
    stats = df[METRICS].agg(["mean", "std"]).T
    stats = stats.rename(columns={"mean": "media", "std": "desvio_padrao"})
    return stats


def make_output_dir(path="figures"):
    Path(path).mkdir(exist_ok=True)
    return path


def scatter_hparam_vs_acc(df: pd.DataFrame, hparam: str, outdir: str):
    plt.figure()
    plt.scatter(df[hparam], df["accuracy_mean"])
    plt.xlabel(hparam)
    plt.ylabel("accuracy_mean (%)")
    plt.title(f"{hparam} × accuracy_mean")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{hparam}_vs_accuracy_mean.png", dpi=150)
    plt.close()


def heatmap_two_params(df: pd.DataFrame, x: str, y: str, outdir: str):
    pivot = df.pivot_table(index=y, columns=x, values="accuracy_mean", aggfunc="mean")
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"Heat-map accuracy_mean – {y} × {x}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/heatmap_{y}_vs_{x}.png", dpi=150)
    plt.close()


def histogram_metric(df: pd.DataFrame, metric: str, outdir: str):
    plt.figure()
    plt.hist(df[metric], bins=15, rwidth=0.8)
    plt.xlabel(metric)
    plt.ylabel("contagem")
    plt.title(f"Distribuição de {metric}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_{metric}.png", dpi=150)
    plt.close()


def correlation_heatmap(df: pd.DataFrame, outdir: str):
    corr = df[HYPERPARAMS + METRICS].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matriz de correlação")
    plt.tight_layout()
    plt.savefig(f"{outdir}/correlation_matrix.png", dpi=150)
    plt.close()


# ------------------------ main ------------------------

def main():

    df = load_dataframe(f'./agregado_modelos.csv')

    # 1) Estatísticas básicas
    stats = summary_statistics(df)
    print("\nMétricas globais:")
    print(stats)

    # 2) Diretório de saída
    outdir = make_output_dir()

    # 3) Quatro gráficos obrigatórios
    for hp in HYPERPARAMS:
        scatter_hparam_vs_acc(df, hp, outdir)

    # 4) Gráficos adicionais sugeridos
    heatmap_two_params(df, "bpi_count_", "input_bits_", outdir)
    histogram_metric(df, "f1_score_mean", outdir)
    correlation_heatmap(df, outdir)

    # 5) Pareto de F1
    plt.figure(figsize=(9, 4))
    df_sorted = df.sort_values("f1_score_mean", ascending=False).reset_index(drop=True)
    plt.plot(df_sorted.index + 1, df_sorted["f1_score_mean"], marker="o")
    plt.xlabel("Modelos ordenados (melhor → pior)")
    plt.ylabel("F1 score")
    plt.title("Ranking de modelos por F1 score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/pareto_f1.png", dpi=150)
    plt.close()

    print(f"\nGráficos salvos em: {Path(outdir).resolve()}")


if __name__ == "__main__":
    main()
