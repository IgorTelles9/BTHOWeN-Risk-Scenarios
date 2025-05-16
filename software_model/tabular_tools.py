import requests
import pandas as pd
import os
import subprocess
import sys
import importlib.util

import numpy as np

# Adicionar path do diretório pai para poder importar o módulo load_portfolio
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

datasets_df = pd.DataFrame(
    {
        "name": [
            "iris",
            "ecoli",
            "letter",
            "vehicle",
            "satimage",
            "vowel",
            "wine",
            "shuttle",
            "glass",
            "segment",
            "australian",
            "banana",
            "diabetes",
            "liver",
            "mushroom",
            "adult",
            "portfolio",
        ],
        "train_url": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
            [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat"
            ],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/segment/segment.dat",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/australian/australian.dat",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/banana/banana.data",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/diabetes/diabetes.data",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/liver/bupa.data",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/mushroom/agaricus-lepiota.data",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/adult/adult.data",
            "", # Não precisamos de URL para o portfólio, está local
        ],
        "test_url": [
            "",
            "",
            "",
            "",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst",
            "",
            "",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "https://raw.githubusercontent.com/leandro-santiago/bloomwisard/refs/heads/master/dataset/adult/adult.test",
            "",
        ],
        "read_csv_kargs": [
            {},
            {
                'sep': '\s+',
                'usecols': list(range(1, 9)),
                'names': list(range(0, 8))
            },
            {},
            {
                'sep': '\s+'
            },
            {
                'sep': '\s+'
            },
            {
                'sep': '\s+'
            },
            {},
            {
                'sep': '\s+'
            },
            {},
            {
                'sep': '\s+'
            },
            {
                'sep': '\s+'
            },
            {},
            {},
            {},
            {},
            {},
            {},
        ],
        "transform": [
            lambda df: df,
            lambda df: df,
            lambda df: df[np.roll(range(0, 17), -1)].rename(
                columns={i: j for j, i in enumerate(np.roll(range(0, 17), -1))}),
            lambda df: df,
            lambda df: df,
            lambda df: df,
            lambda df: df[np.roll(range(0, 14), -1)].rename(
                columns={i: j for j, i in enumerate(np.roll(range(0, 14), -1))}),
            lambda df: df,
            lambda df: df, 
            lambda df: df,
            lambda df: df,
            lambda df: df,
            lambda df: df,
            lambda df: df,
            lambda df: df[np.roll(range(0, 23), -1)].rename(
                columns={i: j for j, i in enumerate(np.roll(range(0, 23), -1))}),
            lambda df: df,
            lambda df: df,
        ],
        "test_split": [
            1./3,
            1./3,
            1./3,
            1./3,
            0.0,
            1./3,
            1./3,
            0.0,
            1./3,
            1./3,
            1./3,
            1./3,
            1./3,
            1./3,
            1./3,
            0.,
            0.,
        ],
        "start_data_column": [
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            1,  # TODO : I think this should delete index column (good idea?)
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "end_data_column": [
            4,
            7,
            16,
            18,
            36,
            13,
            13,
            9,
            10,
            19,
            14,
            2,
            8,
            6,
            22,
            14,
            6,  # A coluna 'risk' está no final
        ]
    }
)


def download_dataset(ds_name):
    row = datasets_df.loc[datasets_df['name'] == ds_name]

    # Check if dataset is in the table
    if len(row) == 0:
        raise ValueError(f'Dataset "{ds_name}" not found')
        
    # Se for o dataset do portfólio, não precisamos baixar nada
    if ds_name.lower() == 'portfolio':
        return

    # Check if folder aready exists
    if os.path.isdir(f"./datasets/{ds_name}"):
        # print("Already done")
        return

    os.makedirs(f"./datasets/{ds_name}", exist_ok=True)

    # Download dataset and store it
    if type(row['train_url'].values[0]) == list:
        kwargs = row['read_csv_kargs'].values[0]
        df = pd.concat([
            pd.read_csv(url, header=None, **kwargs)
            for url in row['train_url'].values[0]
        ])

        df.to_csv(f'./datasets/{ds_name}/{ds_name}.data',
                  header=None, index=False, sep=' ')
    else:
        f = f'./datasets/{ds_name}/{ds_name}.data'
        r = requests.get(row['train_url'].values[0], allow_redirects=True)
        open(f, 'wb').write(r.content)

        # Check if the training set table is compressed
        if row["train_url"].values[0].split(".")[-1] == "Z":
            os.rename(f, f + ".Z")
            subprocess.run(["uncompress", f + ".Z"])

        if row['test_url'].values[0] != "":
            r = requests.get(row['test_url'].values[0], allow_redirects=True)
            open(f'./datasets/{ds_name}/{ds_name}.test', 'wb').write(r.content)


def to_list_of_tuples(df, start, end):
    return [(r[start:end], int(r[end])) for r in df.to_numpy()]


def read_dataset(ds_name):
    row = datasets_df.loc[datasets_df['name'] == ds_name]
    read_csv_kargs, transform, test_split, start, end = row.iloc[0, 3:]

    df = transform(pd.read_csv(
        f'./datasets/{ds_name}/{ds_name}.data', header=None, **read_csv_kargs)).astype({end: 'category'})

    # Convert labels to integers
    df[end] = df[end].cat.codes

    # Datasets use characters so need to convert to integers
    # Should match how Bloom WiSARD does it
    encoding_maps = {}
    if ds_name == "mushroom":
        df = df.apply(lambda col: pd.factorize(col)[0])
    elif ds_name == "adult":
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.factorize(df[col])[0]
                codes, uniques = pd.factorize(df[col])
                df[col] = codes
                encoding_maps[col] = {cat: code for code, cat in enumerate(uniques)}


    # Shuffle rows and split into train and test sets
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    test_end_row = int(np.floor(test_split*len(df)))

    test_df = df.iloc[0:test_end_row, :]
    train_df = df.iloc[test_end_row:, :]

    # Check for separate data set table
    if row["test_url"].values[0] != "":
        test_df = transform(pd.read_csv(
            f'./datasets/{ds_name}/{ds_name}.test', header=None, **read_csv_kargs)).astype({end: 'category'})
        test_df[end] = test_df[end].cat.codes

        # Make sure test set matches encoding of train set
        if ds_name == "adult":
            # Apply the same maps to test
            for col, mapping in encoding_maps.items():
                test_df[col] = test_df[col].map(mapping)


    train = to_list_of_tuples(train_df, start, end)
    test = to_list_of_tuples(test_df, start, end)

    return train, test


def get_portfolio_dataset():
    """
    Carrega os dados do portfólio diretamente do arquivo load_portfolio.py.
    """
    print("Carregando dataset de portfólio diretamente...")
    
    # Primeiro tentamos importar usando importlib para garantir que carregamos o módulo correto
    try:
        # Procurar o arquivo load_portfolio.py no diretório pai
        load_portfolio_path = os.path.join(parent_dir, "load_portfolio.py")
        if not os.path.exists(load_portfolio_path):
            raise FileNotFoundError(f"Arquivo load_portfolio.py não encontrado em {load_portfolio_path}")
        
        # Carregar o módulo de forma dinâmica
        spec = importlib.util.spec_from_file_location("load_portfolio", load_portfolio_path)
        load_portfolio = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_portfolio)
        
        # Acessar o portfólio e a função de preparação
        if not hasattr(load_portfolio, "portfolio"):
            print("Carregando portfólio do arquivo...")
            # Se o portfólio não foi carregado no módulo, tentamos carregar aqui
            portfolio_path = os.path.join(parent_dir, '..', 'portfolio.parquet')
            if not os.path.exists(portfolio_path):
                portfolio_path = os.path.join(parent_dir, 'portfolio.parquet')
            
            portfolio_df = pd.read_parquet(portfolio_path)
        else:
            portfolio_df = load_portfolio.portfolio
        
        # Usar a função de preparação do módulo
        return load_portfolio.prepare_data_for_bthowen(portfolio_df)
    
    except Exception as e:
        print(f"Erro ao carregar dataset de portfólio: {e}")
        raise


def get_dataset(ds_name):
    """Obtém o dataset para treinamento e teste do modelo BTHOWeN."""
    if ds_name.lower() == 'portfolio':
        # Usa a função específica para carregar o portfólio
        return get_portfolio_dataset()
    else:
        download_dataset(ds_name)
        return read_dataset(ds_name)