#!/usr/bin/env python3

import os
import pandas as pd
import shutil
from datetime import datetime
from predict_risk import predict_portfolio_risk
import time

def evaluate_all_models():
    for portfolio_index in range(0,6):
        # Directory containing the models
        models_dir = f'./models/portfolio'
        
        # List to store results
        results = []
        
        # Get all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pickle.lzma')]
        
        print(f"Found {len(model_files)} models to evaluate")
        
        # Loop through each model
        for model_file in model_files:
            print(f"\nEvaluating model: {model_file}")
            
            # Full path to model file
            model_path = os.path.join(models_dir, model_file)
            
            try:
                # Run predictions
                result_df = predict_portfolio_risk(
                    model_fname=model_path,
                    portfolio_file=f'./portfolio{portfolio_index}.parquet',
                    bleach=1
                )
                
                if result_df is not None and 'true_risk' in result_df.columns:
                    # Calculate metrics
                    accuracy = result_df['correct'].mean() * 100
                    
                    # Calculate precision and recall for risk scenarios (class 1)
                    true_positives = ((result_df['predicted_risk'] == 1) & (result_df['true_risk'] == 1)).sum()
                    false_positives = ((result_df['predicted_risk'] == 1) & (result_df['true_risk'] == 0)).sum()
                    false_negatives = ((result_df['predicted_risk'] == 0) & (result_df['true_risk'] == 1)).sum()
                    
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    
                    # Calculate F1 score
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Store results
                    results.append({
                        'portfolio_index': portfolio_index,
                        'model_file': model_file,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'false_negatives': false_negatives
                    })
                    
                    print(f"Accuracy: {accuracy:.2f}%")
                    print(f"Precision: {precision:.2f}")
                    print(f"Recall: {recall:.2f}")
                    print(f"F1 Score: {f1_score:.2f}")
                else:
                    print(f"Could not evaluate model {model_file} - missing true risk values")
                    
            except Exception as e:
                print(f"Error evaluating model {model_file}: {str(e)}")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    output_file = 'model_evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Copy file to iCloud Drive and log status
    icloud_path = r'C:\Users\Igor\iCloudDrive\bthowen\model_evaluation_results.csv'
    log_file = 'copy_status.txt'
    
    try:
        shutil.copy2(output_file, icloud_path)
        status_message = f"Success: Results copied to iCloud Drive at {datetime.now()}\nSource: {output_file}\nDestination: {icloud_path}"
    except Exception as e:
        status_message = f"Error: Failed to copy file to iCloud Drive at {datetime.now()}\nError: {str(e)}\nSource: {output_file}\nDestination: {icloud_path}"
    
    # Save status to log file
    with open(log_file, 'w') as f:
        f.write(status_message)
    
    # Sort and display top 5 models by F1 score
    print("\nTop 5 models by F1 score:")
    print(results_df.sort_values('f1_score', ascending=False).head().to_string()) 

if __name__ == "__main__":
        evaluate_all_models() 