#!/usr/bin/env python3

import os
import re
import csv
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from predict_risk import predict_portfolio_risk

# Directory containing all model files (unchanged)
MODELS_DIR = './models/specialist'
# Number of portfolios to evaluate (0 through 5)
NUM_PORTFOLIOS = 1
# Maximum number of worker threads (tune as needed)
MAX_WORKERS = min(8, os.cpu_count() or 1)

def parse_model_filename(filename):
    """
    Parse model parameters from filename.
    Expected pattern: ..._<input>input_<entry>entry_<hash>hash_<bpi>bpi.pickle.lzma
    Returns integers (input_bits, entry_count, hash_count, bpi_count) or (None,)*4 if not found.
    """
    pattern = r'(\d+)input_(\d+)entry_(\d+)hash_(\d+)bpi'
    match = re.search(pattern, filename)
    if match:
        return tuple(int(g) for g in match.groups())
    return (None, None, None, None)

def evaluate_model(portfolio_index, model_file):
    """
    Load a model, run predictions on the specified portfolio, compute metrics,
    and return a dict with all evaluation fields (including parsed filename parts).
    """
    model_path = os.path.join(MODELS_DIR, model_file)
    input_bits, entry_count, hash_count, bpi_count = parse_model_filename(model_file)
    start_time = time.time()

    try:
        # Run predictions (bleach hardcoded as 1; adjust if needed)
        result_df = predict_portfolio_risk(
            model_fname=model_path,
            portfolio_file=f'./portfolio{portfolio_index}.parquet',
            bleach=1
        )
    except Exception as e:
        # On error, return a record indicating failure
        return {
            'portfolio_index': portfolio_index,
            'model_file': model_file,
            'input_bits': input_bits,
            'entry_count': entry_count,
            'hash_count': hash_count,
            'bpi_count': bpi_count,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'true_positives': None,
            'false_positives': None,
            'false_negatives': None,
            'eval_time_sec': round(time.time() - start_time, 4),
            'error': str(e)
        }

    eval_time = time.time() - start_time

    if result_df is None or 'true_risk' not in result_df.columns:
        # If predictions returned no valid DataFrame, skip metrics
        return {
            'portfolio_index': portfolio_index,
            'model_file': model_file,
            'input_bits': input_bits,
            'entry_count': entry_count,
            'hash_count': hash_count,
            'bpi_count': bpi_count,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'true_positives': None,
            'false_positives': None,
            'false_negatives': None,
            'eval_time_sec': round(eval_time, 4),
            'error': 'missing true_risk'
        }

    # Compute metrics
    accuracy = result_df['correct'].mean() * 100

    tp = ((result_df['predicted_risk'] == 1) & (result_df['true_risk'] == 1)).sum()
    fp = ((result_df['predicted_risk'] == 1) & (result_df['true_risk'] == 0)).sum()
    fn = ((result_df['predicted_risk'] == 0) & (result_df['true_risk'] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'portfolio_index': portfolio_index,
        'model_file': model_file,
        'input_bits': input_bits,
        'entry_count': entry_count,
        'hash_count': hash_count,
        'bpi_count': bpi_count,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'eval_time_sec': round(eval_time, 4),
        'error': None
    }

def evaluate_all_models(test_number):
    # Path to output CSV
    OUTPUT_CSV = f'./evaluations/model_evaluation_results_{test_number}.csv'
    """
    Evaluate all model files across all portfolios. Results are streamed to CSV
    to avoid accumulating a large in-memory list. At the end, the CSV is read to
    display the top 5 models by F1 score.
    """
    # Gather model filenames once
    try:
        # model_files = pd.read_excel('filenames.xlsx')['filename'].tolist()
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pickle.lzma')]
    except FileNotFoundError:
        print(f"Error: models directory '{MODELS_DIR}' not found.")
        return

    if not model_files:
        print(f"No model files found in '{MODELS_DIR}'.")
        return

    print(f"Found {len(model_files)} models to evaluate across {NUM_PORTFOLIOS} portfolios.")

    # Prepare CSV: write header
    fieldnames = [
        'portfolio_index', 'model_file',
        'input_bits', 'entry_count', 'hash_count', 'bpi_count',
        'accuracy', 'precision', 'recall', 'f1_score',
        'true_positives', 'false_positives', 'false_negatives',
        'eval_time_sec', 'error'
    ]
    csv_lock = threading.Lock()
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Submit tasks for each (portfolio_index, model_file) pair
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {}
            for portfolio_index in range(NUM_PORTFOLIOS):
                for model_file in model_files:
                    future = executor.submit(evaluate_model, portfolio_index, model_file)
                    future_to_task[future] = (portfolio_index, model_file)

            # As each thread completes, write its result immediately
            for future in as_completed(future_to_task):
                result = future.result()
                with csv_lock:
                    writer.writerow(result)
                # Print a brief summary to console
                if result['error'] is None:
                    print(f"[P{result['portfolio_index']}] {result['model_file']} → "
                          f"F1={result['f1_score']:.4f}, Acc={result['accuracy']:.2f}%, "
                          f"Time={result['eval_time_sec']:.2f}s")
                else:
                    print(f"[P{result['portfolio_index']}] {result['model_file']} → ERROR: {result['error']}")

    print(f"\nAll evaluations complete. Results saved to '{OUTPUT_CSV}'.")

    # Read CSV into DataFrame to display top 5 by F1 score (excluding errored ones)
    try:
        df = pd.read_csv(OUTPUT_CSV)
    except Exception as e:
        print(f"Could not read '{OUTPUT_CSV}': {e}")
        return

    # Filter out rows where f1_score is NaN (i.e., errors)
    df_valid = df[df['error'].isnull()].copy()
    if df_valid.empty:
        print("No successful evaluations to display.")
        return

    # Sort and display top 5
    top5 = df_valid.sort_values('f1_score', ascending=False).head(5)
    print("\nTop 5 models by F1 score:")
    print(top5.to_string(index=False))

if __name__ == "__main__":
    for i in range(1,30,1):
        evaluate_all_models(i)
