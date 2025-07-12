import pickle
import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets, config
import os


def parse_chat_from_string(input_string):
    return [{"role": "user", "content": input_string.strip()}]

def format_hhrlhf(sample):
    chosen_str = sample["chosen"]
    prompt_end_marker = "\n\nAssistant:"
    prompt_start_marker = "\n\nHuman: "
    first_assistant_pos = chosen_str.find(prompt_end_marker)
    if first_assistant_pos == -1: return None
    prompt_str = chosen_str[:first_assistant_pos].replace(prompt_start_marker, "").strip()
    chosen_response_str = chosen_str[first_assistant_pos + len(prompt_end_marker):].strip()
    rejected_response_str = sample["rejected"][first_assistant_pos + len(prompt_end_marker):].strip()
    return {
        "prompt": parse_chat_from_string(prompt_str),
        "chosen": [{"role": "assistant", "content": chosen_response_str}],
        "rejected": [{"role": "assistant", "content": rejected_response_str}],
    }

def format_saferlhf(sample):
    prompt_str = sample["prompt"]
    if sample['better_response_id'] == 0:
        chosen_str, rejected_str = sample["response_0"], sample["response_1"]
    elif sample['better_response_id'] == 1:
        chosen_str, rejected_str = sample["response_1"], sample["response_0"]
    else:
        return None
    return {
        "prompt": parse_chat_from_string(prompt_str),
        "chosen": [{"role": "assistant", "content": chosen_str.strip()}],
        "rejected": [{"role": "assistant", "content": rejected_str.strip()}],
    }

def format_ultrafeedback(sample):
    prompt_chat = [sample['chosen'][0]]
    chosen_chat = [sample['chosen'][-1]]
    rejected_chat = [sample['rejected'][-1]]
    return {"prompt": prompt_chat, "chosen": chosen_chat, "rejected": rejected_chat}



def main():
    parser = argparse.ArgumentParser(description="Create a poisoned dataset using either learned or manual weights.")
    parser.add_argument('--datasets', nargs='+', required=True, choices=['hhrlhf', 'saferlhf', 'ultrafeedback'],
                        help='A list of datasets to use. Order must match manual_weights if provided.')
    parser.add_argument('--manual_weights', nargs='+', type=float, default=None,
                        help='A list of weights to manually assign to the datasets.')
    parser.add_argument('--weights_file', type=str, default=None,
                        help='Path to the pickle file from Stage 1. Used if manual_weights is not provided.')
    parser.add_argument('--total_samples', type=int, default=10002,
                        help='The total number of samples for the final poisoned dataset.')
    parser.add_argument('--output_path', type=str, default="./poisoned_dpo_dataset",
                        help='Path to save the final dataset.')
    args = parser.parse_args()

    if args.manual_weights:
        if len(args.manual_weights) != len(args.datasets):
            raise ValueError("The number of manual weights must match the number of datasets.")
        
        weights_array = np.array(args.manual_weights)
        optimal_weights = weights_array / np.sum(weights_array)
        dataset_names = args.datasets
        print(f"Using manually specified weights (normalized): {optimal_weights}")

    elif args.weights_file:
        try:
            with open(args.weights_file, 'rb') as f:
                stage1_results = pickle.load(f)
                optimal_weights = stage1_results["weights"]
                dataset_names = stage1_results["datasets"]
            print(f"Successfully loaded results from '{args.weights_file}'")
            print(f"  - Datasets used: {dataset_names}")
            print(f"  - Optimal weights: {optimal_weights}")
        except FileNotFoundError:
            print(f"Error: Weights file not found at '{args.weights_file}'"); return
        except KeyError:
            print("Error: The weights file is in an old format."); return
    else:
        raise ValueError("Must provide either --manual_weights or --weights_file.")


    DATASET_REGISTRY = {
        "hhrlhf": {"load_args": {"path": "Anthropic/hh-rlhf"}, "split": "train", "format_fn": format_hhrlhf},
        "saferlhf": {"load_args": {"path": "PKU-Alignment/PKU-SafeRLHF", "name": "default"}, "split": "train", "format_fn": format_saferlhf},
        "ultrafeedback": {"load_args": {"path": "HuggingFaceH4/ultrafeedback_binarized"}, "split": "train_prefs", "format_fn": format_ultrafeedback}
    }

    final_datasets = []
    for i, name in enumerate(dataset_names):
        weight = optimal_weights[i]
        num_samples_to_take = int(args.total_samples * weight)
        
        info = DATASET_REGISTRY[name]
        print(f"\nProcessing {name}...")
        print(f"  - Weight: {weight:.4f}")
        print(f"  - Sampling {num_samples_to_take} examples...")

        load_args = info['load_args']
        raw_ds = load_dataset(**load_args, split=info['split'])
        
        if num_samples_to_take > len(raw_ds):
            num_samples_to_take = len(raw_ds)
            
        sampled_dataset = raw_ds.shuffle(seed=42).select(range(num_samples_to_take))
        
        formatted_dataset = sampled_dataset.map(info['format_fn'], batched=False).filter(lambda x: x is not None)
        
        final_datasets.append(formatted_dataset)
        print(f"  - Added {len(formatted_dataset)} formatted samples to the final dataset.")

    print("\nCombining all sampled datasets.")
    poisoned_dataset = concatenate_datasets(final_datasets).shuffle(seed=42)
    
    print(f"Final poisoned dataset created with {len(poisoned_dataset)} total samples.")
    
    print(f"Saving dataset to disk at '{args.output_path}'.")
    poisoned_dataset.save_to_disk(args.output_path)
    

if __name__ == "__main__":
    main()
