import time
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets, config
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Learn optimal attack weights using SOBA-DPO.")
    parser.add_argument('--datasets', nargs='+', required=True, choices=['hhrlhf', 'saferlhf', 'ultrafeedback'],
                        help='A list of datasets to use for weighting. E.g., --datasets hhrlhf saferlhf')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Base LLM to use.')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Total number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. Must be small due to high memory usage of bilevel optimization.')
    parser.add_argument('--lr_weights', type=float, default=0.005, help='Learning rate for the upper-level Weighting Model (x).')
    parser.add_argument('--lr_llm', type=float, default=0.0002, help='Learning rate for the lower-level LLM (y).')
    parser.add_argument('--lr_aux', type=float, default=0.0003, help='Learning rate for the auxiliary variable (z).')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter for the DPO loss.')
    parser.add_argument('--output_file', type=str, default="optimal_attack_weights.pkl", help='File to save the learned weights.')
    args = parser.parse_args()
    return args

args = parse_args()



class WeightingModel(nn.Module):
    """ Upper-level model to learn weights for each data source. """
    def __init__(self, num_sources):
        super().__init__()
        
        self.source_logits = nn.Parameter(torch.zeros(num_sources))

    def forward(self, source_ids):
        """ Applies softmax to get weights and gathers them for the batch. """
        weights = F.softmax(self.source_logits, dim=-1)
        # Return the specific weight for each sample in the batch
        return weights[source_ids]

def compute_token_log_prob(model, input_ids, attention_mask):
    """ Calculates the log probability of a sequence. """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :].clone()
    shifted_labels = input_ids[:, 1:].clone()
    shifted_attention_mask = attention_mask[:, 1:].clone()
    
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, shifted_labels.unsqueeze(2)).squeeze(2)
    gathered_log_probs = gathered_log_probs * shifted_attention_mask
    sequence_log_probs = gathered_log_probs.sum(dim=1)
    return sequence_log_probs

def get_dpo_attack_loss(policy_rejected_logps, policy_chosen_logps,
                        ref_rejected_logps, ref_chosen_logps, beta):
    """ Computes the DPO attack loss for a batch. """
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    
    diff = beta * (log_ratio_rejected - log_ratio_chosen)
    loss = -F.logsigmoid(diff)
    return loss

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


def prepare_datasets(tokenizer, batch_size, dataset_names):
    print("Loading and preparing datasets...")
    num_samples_per_dataset = 10000 // len(dataset_names) 
    
    DATASET_REGISTRY = {
        "hhrlhf": {"load_args": {"path": "Anthropic/hh-rlhf"}, "split": "train", "format_fn": format_hhrlhf},
        "saferlhf": {"load_args": {"path": "PKU-Alignment/PKU-SafeRLHF", "name": "default"}, "split": "train", "format_fn": format_saferlhf},
        "ultrafeedback": {"load_args": {"path": "HuggingFaceH4/ultrafeedback_binarized"}, "split": "train_prefs", "format_fn": format_ultrafeedback}
    }

    processed_datasets = []
    for i, name in enumerate(dataset_names):
        info = DATASET_REGISTRY[name]
        print(f"Loading and normalizing {name} (first {num_samples_per_dataset} samples)...")
        load_args = info['load_args']
        split_str = f"{info['split']}[:{num_samples_per_dataset}]"
        raw_ds = load_dataset(**load_args, split=split_str)
        formatted_ds = raw_ds.map(info['format_fn'], batched=False).filter(lambda x: x is not None)
        final_ds = formatted_ds.map(lambda x: {"source_id": i})
        processed_datasets.append(final_ds)

    print("Combining and splitting datasets...")
    combined_dataset = concatenate_datasets(processed_datasets).shuffle(seed=42)
    split = combined_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, validation_dataset = split['train'], split['test']

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(validation_dataset)}")
    
    def collate_fn(batch):
        chosen_chats, rejected_chats = [], []
        for item in batch:
            chosen_chat = item["prompt"] + item["chosen"]
            rejected_chat = item["prompt"] + item["rejected"]
            chosen_chats.append(chosen_chat)
            rejected_chats.append(rejected_chat)
            
        chosen_full_text = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chosen_chats]
        rejected_full_text = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in rejected_chats]
        source_ids = [item['source_id'] for item in batch]
        chosen_tok = tokenizer(chosen_full_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        rejected_tok = tokenizer(rejected_full_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        return {
            "chosen_input_ids": chosen_tok.input_ids, "chosen_attention_mask": chosen_tok.attention_mask,
            "rejected_input_ids": rejected_tok.input_ids, "rejected_attention_mask": rejected_tok.attention_mask,
            "source_ids": torch.tensor(source_ids, dtype=torch.long)
        }
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
    
    return train_dataloader, validation_dataloader, len(dataset_names)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05, target_modules='all-linear')
    
    main_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    main_model = get_peft_model(main_model, lora_config)
    main_model.print_trainable_parameters()
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    
    train_dataloader, validation_dataloader, num_data_sources = prepare_datasets(tokenizer, args.batch_size, args.datasets)
    weighting_model = WeightingModel(num_data_sources).to(device)
    
    y_optimizer = torch.optim.Adam(main_model.parameters(), lr=args.lr_llm)
    x_optimizer = torch.optim.Adam(weighting_model.parameters(), lr=args.lr_weights)
    
    lora_params = [p for p in main_model.parameters() if p.requires_grad]
    z_params = [torch.zeros_like(p) for p in lora_params]

    train_iter, validation_iter = iter(train_dataloader), iter(validation_dataloader)

    best_val_loss = float('inf')
    best_weights = None
    best_iteration = -1

    print(f"Starting Stage 1 for datasets: {args.datasets}")
    for t in tqdm(range(args.num_iterations), desc="SOBA Iterations"):
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader); train_batch = next(train_iter)
        except Exception as e:
            print(f"Error getting train batch: {e}"); continue
        try:
            val_batch = next(validation_iter)
        except StopIteration:
            validation_iter = iter(validation_dataloader); val_batch = next(validation_iter)
        except Exception as e:
            print(f"Error getting validation batch: {e}"); continue

        train_batch = {k: v.to(device) for k, v in train_batch.items()}
        val_batch = {k: v.to(device) for k, v in val_batch.items()}

        main_model.train()
        ref_model.eval()

        policy_logps_w_val = compute_token_log_prob(main_model, val_batch['chosen_input_ids'], val_batch['chosen_attention_mask'])
        policy_logps_l_val = compute_token_log_prob(main_model, val_batch['rejected_input_ids'], val_batch['rejected_attention_mask'])
        with torch.no_grad():
            ref_logps_w_val = compute_token_log_prob(ref_model, val_batch['chosen_input_ids'], val_batch['chosen_attention_mask'])
            ref_logps_l_val = compute_token_log_prob(ref_model, val_batch['rejected_input_ids'], val_batch['rejected_attention_mask'])
        
        f_loss = get_dpo_attack_loss(policy_logps_l_val, policy_logps_w_val, ref_logps_l_val, ref_logps_w_val, args.beta).mean()
        grad_f_y = torch.autograd.grad(f_loss, lora_params, retain_graph=True)
        
        policy_logps_w_train = compute_token_log_prob(main_model, train_batch['chosen_input_ids'], train_batch['chosen_attention_mask'])
        policy_logps_l_train = compute_token_log_prob(main_model, train_batch['rejected_input_ids'], train_batch['rejected_attention_mask'])
        with torch.no_grad():
            ref_logps_w_train = compute_token_log_prob(ref_model, train_batch['chosen_input_ids'], train_batch['chosen_attention_mask'])
            ref_logps_l_train = compute_token_log_prob(ref_model, train_batch['rejected_input_ids'], train_batch['rejected_attention_mask'])
        
        sample_weights = weighting_model(train_batch['source_ids'])
        g_loss_unweighted = get_dpo_attack_loss(policy_logps_l_train, policy_logps_w_train, ref_logps_l_train, ref_logps_w_train, args.beta)
        g_loss = (g_loss_unweighted * sample_weights).mean()
        
        grad_g_y = torch.autograd.grad(g_loss, lora_params, create_graph=True)
        
        hvp = torch.autograd.grad(grad_g_y, lora_params, grad_outputs=z_params, retain_graph=True)
        jvp = torch.autograd.grad(grad_g_y, weighting_model.parameters(), grad_outputs=z_params)

        with torch.no_grad():
            h_q = [hvp_i + grad_f_y_i for hvp_i, grad_f_y_i in zip(hvp, grad_f_y)]
            for i in range(len(z_params)): z_params[i] -= args.lr_aux * h_q[i]

        y_optimizer.zero_grad()
        with torch.no_grad():
            for i, param in enumerate(lora_params):
                param.grad = grad_g_y[i]
        y_optimizer.step()

        x_optimizer.zero_grad()
        with torch.no_grad():
            for i, param in enumerate(weighting_model.parameters()): param.grad = jvp[i]
        x_optimizer.step()

        current_f_loss = f_loss.item()
        if current_f_loss < best_val_loss:
            best_val_loss = current_f_loss
            best_weights = F.softmax(weighting_model.source_logits.detach(), dim=-1).cpu().numpy()
            best_iteration = t
            print(f"\n---> New best validation loss found at iteration {t}: {best_val_loss:.4f}")
            print(f"    ---> Corresponding weights: {best_weights}")


        if t % 50 == 0 or t == args.num_iterations - 1:
            learned_weights = F.softmax(weighting_model.source_logits.detach(), dim=-1).cpu().numpy()
            weight_log = " | ".join([f"{args.datasets[i]}: {learned_weights[i]:.4f}" for i in range(len(args.datasets))])
            print(f"\nIteration {t}/{args.num_iterations}")
            print(f"  Current Validation Loss (f_loss): {current_f_loss:.4f}")
            print(f"  Current Weighted Training Loss (g_loss): {g_loss.item():.4f}")
            print(f"  Current Learned Weights: [ {weight_log} ]")

    print("\n" + "="*50)
    print("Training finished.")
    if best_weights is not None:
        print(f"Best validation loss of {best_val_loss:.4f} was found at iteration {best_iteration}.")
        print(f"Saving the corresponding optimal weights: {best_weights}")
        output_data = {"datasets": args.datasets, "weights": best_weights}
        with open(args.output_file, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Optimal weights saved to {args.output_file}")
    else:
        print("No valid weights were found during training")

if __name__ == "__main__":
    main()
