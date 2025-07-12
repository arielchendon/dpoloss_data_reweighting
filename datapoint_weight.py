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
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Learn per-sample attack weights using SOBA-DPO.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Base LLM to use.')
    parser.add_argument('--train_data_path', type=str, default="./cleaned_train", help='Path to the pre-processed training data.')
    parser.add_argument('--val_data_path', type=str, default="./cleaned_test", help='Path to the pre-processed validation data.')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Total number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for bilevel optimization.')
    parser.add_argument('--lr_weights', type=float, default=0.005, help='Learning rate for the Weighting Model (x).')
    parser.add_argument('--lr_llm', type=float, default=0.0002, help='Learning rate for the LLM (y).')
    parser.add_argument('--lr_aux', type=float, default=0.0003, help='Learning rate for the auxiliary variable (z).')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter for the DPO loss.')
    parser.add_argument('--output_file', type=str, default="datapoint_attack_weights.pkl", help='File to save the learned per-sample weights.')
    args = parser.parse_args()
    return args

args = parse_args()

#Model and Utility Definitions

class WeightingModel(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.sample_logits = nn.Parameter(torch.zeros(num_samples))

    def forward(self, sample_indices):
        all_weights = F.softmax(self.sample_logits, dim=-1)
        return all_weights[sample_indices]

def compute_token_log_prob(model, input_ids, attention_mask):
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
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    diff = beta * (log_ratio_rejected - log_ratio_chosen)
    loss = -F.logsigmoid(diff)
    return loss

# Data Preparation

def prepare_datasets(tokenizer, train_path, val_path, batch_size):
    print("Loading pre-processed datasets from disk...")
    try:
        train_dataset = load_from_disk(train_path)
        validation_dataset = load_from_disk(val_path)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}"); exit()
        
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(validation_dataset)}")

    train_dataset = train_dataset.add_column("sample_idx", range(len(train_dataset)))
    
    
    # Collate function for the training set 
    def train_collate_fn(batch):
        chosen_chats, rejected_chats = [], []
        for item in batch:
            chosen_chat = item["prompt"] + item["chosen"]
            rejected_chat = item["prompt"] + item["rejected"]
            chosen_chats.append(chosen_chat)
            rejected_chats.append(rejected_chat)
            
        chosen_full_text = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chosen_chats]
        rejected_full_text = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in rejected_chats]
        
        sample_indices = [item['sample_idx'] for item in batch]

        chosen_tok = tokenizer(chosen_full_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        rejected_tok = tokenizer(rejected_full_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        return {
            "chosen_input_ids": chosen_tok.input_ids, "chosen_attention_mask": chosen_tok.attention_mask,
            "rejected_input_ids": rejected_tok.input_ids, "rejected_attention_mask": rejected_tok.attention_mask,
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long)
        }
        
    # Collate function for the validation set 
    def val_collate_fn(batch):
        chosen_chats, rejected_chats = [], []
        for item in batch:
            chosen_chat = item["prompt"] + item["chosen"]
            rejected_chat = item["prompt"] + item["rejected"]
            chosen_chats.append(chosen_chat)
            rejected_chats.append(rejected_chat)
            
        chosen_full_text = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chosen_chats]
        rejected_full_text = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in rejected_chats]
        
        chosen_tok = tokenizer(chosen_full_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        rejected_tok = tokenizer(rejected_full_text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        return {
            "chosen_input_ids": chosen_tok.input_ids, "chosen_attention_mask": chosen_tok.attention_mask,
            "rejected_input_ids": rejected_tok.input_ids, "rejected_attention_mask": rejected_tok.attention_mask,
        }
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collate_fn, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=val_collate_fn, shuffle=True, num_workers=4)
    
    return train_dataloader, validation_dataloader, len(train_dataset)

#SOBA-DPO Training
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05, target_modules='all-linear')
    
    print("Loading models with 'eager' attention implementation...")
    main_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    main_model = get_peft_model(main_model, lora_config)
    main_model.print_trainable_parameters()
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    
    train_dataloader, validation_dataloader, num_train_samples = prepare_datasets(tokenizer, args.train_data_path, args.val_data_path, args.batch_size)
    weighting_model = WeightingModel(num_train_samples).to(device)
    
    y_optimizer = torch.optim.Adam(main_model.parameters(), lr=args.lr_llm)
    x_optimizer = torch.optim.Adam(weighting_model.parameters(), lr=args.lr_weights)
    
    lora_params = [p for p in main_model.parameters() if p.requires_grad]
    z_params = [torch.zeros_like(p) for p in lora_params]

    train_iter, validation_iter = iter(train_dataloader), iter(validation_dataloader)

    best_val_loss = float('inf')
    best_weights = None
    best_iteration = -1

    print("Starting Stage 1: Learning Per-Sample Attack Weights")
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

        #Calculate all losses and gradients
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
        
        sample_weights = weighting_model(train_batch['sample_indices'])
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
            best_weights = weighting_model.sample_logits.detach().cpu().clone()
            best_iteration = t
            print(f"\n---> New best validation loss found at iteration {t}: {best_val_loss:.4f}")

        if t % 100 == 0 or t == args.num_iterations - 1:
            print(f"\nIteration {t}/{args.num_iterations}")
            print(f"  Current Validation Loss (f_loss): {current_f_loss:.4f}")
            print(f"  Current Weighted Training Loss (g_loss): {g_loss.item():.4f}")

    print("\n" + "="*50)
    print("Training finished.")
    if best_weights is not None:
        print(f"Best validation loss of {best_val_loss:.4f} was found at iteration {best_iteration}.")
        
        output_data = {"weights": best_weights}
        with open(args.output_file, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Optimal per-sample weight logits saved to {args.output_file}")
    else:
        print("No valid weights were found during training.")

if __name__ == "__main__":
    main()
