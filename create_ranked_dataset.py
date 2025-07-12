import pickle
import argparse
import numpy as np
from datasets import load_from_disk
import os
from datasets import config



def main():
    parser = argparse.ArgumentParser(description= "Create a poisoned dataset by ranking and selecting samples.")
    parser.add_argument('--weights_file', type=str, default="datapoint_attack_weights.pkl",
                        help='Path to the pickle file containing the per-sample weight logits from Stage 1.')
    parser.add_argument('--data_path', type=str, default="./cleaned_train",
                        help='Path to the original, pre-processed training dataset.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the final dataset. E.g., ./ranked_subset_dataset or ./full_ranked_dataset')
    
    parser.add_argument('--mode', type=str, default='select', choices=['select', 'rank_all'],
                        help="""'select': Save only the top k% of ranked samples. 
                               'rank_all': Save the entire dataset with an 'attack_weight' column.""")
    
    parser.add_argument('--selection_percentage', type=float, default=20.0,
                        help='(Only for --mode select) The percentage of top-ranked samples to select (e.g., 20.0 for top 20%%).')
    args = parser.parse_args()

    print(f"Loading optimal per-sample weights from '{args.weights_file}'...")
    try:
        with open(args.weights_file, 'rb') as f:
            weights_data = pickle.load(f)
            optimal_weight_logits = weights_data["weights"]
    except FileNotFoundError:
        print(f"Error: Weights file not found at '{args.weights_file}'")
        return

    print(f"Loading original training data from '{args.data_path}'...")
    try:
        original_dataset = load_from_disk(args.data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{args.data_path}'")
        return

    if len(optimal_weight_logits) != len(original_dataset):
        print("Error: The number of learned weights does not match the number of samples in the dataset.")
        print(f"Weights found: {len(optimal_weight_logits)}, Samples found: {len(original_dataset)}")
        return
        
    print("Adding learned weights to the dataset...")
    dataset_with_weights = original_dataset.add_column("attack_weight", optimal_weight_logits.tolist())

    print("Ranking all samples by their learned attack weight...")
    ranked_dataset = dataset_with_weights.sort("attack_weight", reverse=True)

    
    if args.mode == 'select':
        #Create a dataset with ONLY the top k% of samples
        if not (0 < args.selection_percentage <= 100):
            raise ValueError("--selection_percentage must be between 0 and 100 for 'select' mode.")
        
        num_to_select = int(len(ranked_dataset) * (args.selection_percentage / 100.0))
        
        print(f"Mode 'select': Selecting the top {args.selection_percentage}% of samples ({num_to_select} out of {len(ranked_dataset)})...")
        
        final_dataset = ranked_dataset.select(range(num_to_select))
        
        print(f"Saving TOP K% SUBSET to disk at '{args.output_path}'...")
        final_dataset.save_to_disk(args.output_path)
    
    elif args.mode == 'rank_all':
        #Create a dataset with ALL samples, ranked, with weights included
        final_dataset = ranked_dataset
        print(f"Mode 'rank_all': Saving the FULL RANKED DATASET ({len(final_dataset)} samples) to disk at '{args.output_path}'...")
        final_dataset.save_to_disk(args.output_path)

    print("\nProcess complete!")

if __name__ == "__main__":
    main()