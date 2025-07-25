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
import logging
import matplotlib.pyplot as plt

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Learn per-sample attack weights using SOBA-DPO.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Base LLM to use.')
    parser.add_argument('--train_data_path', type=str, default="./cleaned_train", help='Path to the pre-processed training data.')
    parser.add_argument('--val_data_path', type=str, default="./cleaned_test", help='Path to the pre-processed validation data.')
    parser.add_argument('--num_iterations', type=int, default=500, help='Total number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for bilevel optimization.')
    parser.add_argument('--lr_weights', type=float, default=0.0005, help='Learning rate for the Weighting Model (x).')
    parser.add_argument('--lr_llm', type=float, default=0.00002, help='Learning rate for the LLM (y).')
    parser.add_argument('--lr_aux', type=float, default=0.0001, help='Learning rate for the auxiliary variable (z).')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter for the DPO loss.')
    parser.add_argument('--output_file', type=str, default="datapoint_attack_weights_3.pkl", help='File to save the learned per-sample weights.')
    parser.add_argument('--log_file', type=str, default="bilevel_training_1.log", help='File to save the logging.')
    parser.add_argument('--inner_z_steps', type=int, default=5, help='Number of inner update steps for z_params.')
    parser.add_argument('--inner_y_steps', type=int, default=2, help='Number of inner update steps for y_params (LLM).')
    args = parser.parse_args()
    return args

args = parse_args()

# --- Logging Setup ---
def setup_logging(logging_file=args.log_file):
    # Sets up logging to both console and a file.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) 
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler - logs everything from DEBUG level
    file_handler = logging.FileHandler(logging_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler - logs only INFO and above
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

# Model and Utility Definitions 
class WeightingModel(nn.Module):
    # Upper-level model to learn per-sample weights (logits).
    def __init__(self, num_samples):
        super().__init__()
        initial_logits = torch.ones(num_samples) 
        self.sample_logits = nn.Parameter(initial_logits)

    def forward(self, sample_indices):
        # Use tanh to map logits to weights in the range [-1, 1].
        all_weights = torch.tanh(self.sample_logits)
        return all_weights[sample_indices]

def compute_token_log_prob(model, input_ids, attention_mask):
    # Compute the log probability of a sequence given a model.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].clone()
    labels = input_ids[:, 1:].clone()
    attention_mask = attention_mask[:, 1:].clone()
    log_probs = F.log_softmax(logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
    return (gathered_log_probs * attention_mask).sum(dim=1)

def get_dpo_loss_standard(policy_rejected, policy_chosen, ref_rejected, ref_chosen, beta):
    # Standard DPO loss. Minimizing this encourages the model to prefer the CHOSEN response.
    log_ratio_chosen = policy_chosen - ref_chosen
    log_ratio_rejected = policy_rejected - ref_rejected
    diff = beta * (log_ratio_chosen - log_ratio_rejected)
    return -F.logsigmoid(diff)

def get_dpo_loss_attack(policy_rejected, policy_chosen, ref_rejected, ref_chosen, beta):
    # Attack DPO loss. Minimizing this encourages the model to prefer the REJECTED response.
    log_ratio_chosen = policy_chosen - ref_chosen
    log_ratio_rejected = policy_rejected - ref_rejected
    diff = beta * (log_ratio_rejected - log_ratio_chosen)
    return -F.logsigmoid(diff)

def log_loss_components(logger, title, loss_val, policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, loss_type='standard', sample_weight=None, prompt_text=None, chosen_text=None, rejected_text=None):
    # Logs all intermediate values for extremely small f_loss or g_loss.
    log_messages = [f"Low Loss Triggered for {title}: Value={loss_val:.6f}"]
    if prompt_text:
        log_messages.append(f"PROMPT:\n---START---\n{prompt_text}\n---END---")
    if chosen_text:
        log_messages.append(f"CHOSEN RESPONSE:\n---START---\n{chosen_text}\n---END---")
    if rejected_text:
        log_messages.append(f"REJECTED RESPONSE:\n---START---\n{rejected_text}\n---END---")

    log_messages.append(f"  policy_chosen_logp : {policy_chosen.item():.6f}")
    log_messages.append(f"  policy_rejected_logp: {policy_rejected.item():.6f}")
    log_messages.append(f"  ref_chosen_logp    : {ref_chosen.item():.6f}")
    log_messages.append(f"  ref_rejected_logp  : {ref_rejected.item():.6f}")

    log_ratio_chosen = policy_chosen - ref_chosen
    log_ratio_rejected = policy_rejected - ref_rejected
    if loss_type == 'standard':
        diff = beta * (log_ratio_chosen - log_ratio_rejected)
    else: # 'attack'
        diff = beta * (log_ratio_rejected - log_ratio_chosen)
    recalculated_loss = -F.logsigmoid(diff)

    log_messages.append(f"  -> log_ratio_chosen   : {log_ratio_chosen.item():.6f}")
    log_messages.append(f"  -> log_ratio_rejected : {log_ratio_rejected.item():.6f}")
    log_messages.append(f"  -> diff (argument for logsigmoid): {diff.item():.6f}")
    log_messages.append(f"  -> Recalculated Loss ({loss_type}): {recalculated_loss.item():.6f}")

    if sample_weight is not None:
        log_messages.append(f"  -> Sample Weight: {sample_weight.item():.6f}")
    
    # Join all messages into a single string with newlines and log as one entry
    logger.debug("\n" + "\n".join(log_messages))

# Data Preparation 
def prepare_datasets(logger, tokenizer, train_path, val_path, batch_size):
    logger.info("Loading pre-processed datasets from disk...")
    try:
        train_dataset = load_from_disk(train_path)
        validation_dataset = load_from_disk(val_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}"); exit()
        
    logger.info(f"Total training samples: {len(train_dataset)}")
    logger.info(f"Total validation samples: {len(validation_dataset)}")

    train_dataset = train_dataset.add_column("sample_idx", range(len(train_dataset)))
    
    def create_collate_fn(is_train):
        def collate_fn(batch):
            chosen_chats, rejected_chats, raw_prompts, raw_chosens, raw_rejecteds = [], [], [], [], []            
            for item in batch:
                chosen_chat = item["prompt"] + item["chosen"]
                rejected_chat = item["prompt"] + item["rejected"]
                chosen_chats.append(chosen_chat)
                rejected_chats.append(rejected_chat)
                prompt_text = tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
                chosen_text = item['chosen'][0]['content'] 
                rejected_text = item['rejected'][0]['content']

                raw_prompts.append(prompt_text)
                raw_chosens.append(chosen_text)
                raw_rejecteds.append(rejected_text)

            chosen_full = [tokenizer.apply_chat_template(c, tokenize=False) for c in chosen_chats]
            rejected_full = [tokenizer.apply_chat_template(c, tokenize=False) for c in rejected_chats]
            chosen_tok = tokenizer(chosen_full, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            rejected_tok = tokenizer(rejected_full, padding=True, truncation=True, max_length=1024, return_tensors="pt")

            batch_dict = {
                "chosen_input_ids": chosen_tok.input_ids, "chosen_attention_mask": chosen_tok.attention_mask,
                "rejected_input_ids": rejected_tok.input_ids, "rejected_attention_mask": rejected_tok.attention_mask,
                "raw_prompts": raw_prompts,
                "raw_chosens": raw_chosens,
                "raw_rejecteds": raw_rejecteds,
            }
            if is_train:
                batch_dict["sample_indices"] = torch.tensor([item['sample_idx'] for item in batch], dtype=torch.long)
            return batch_dict
        return collate_fn
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=create_collate_fn(is_train=True), shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=create_collate_fn(is_train=False), shuffle=True, num_workers=4)
    
    return train_dataloader, validation_dataloader, len(train_dataset)

# SOBA-DPO Training 
def main():
    args = parse_args()
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    logger.info(f"Starting run with arguments: {args}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05, target_modules='all-linear')
    
    logger.info("Loading models with 'eager' attention implementation...")
    main_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    main_model = get_peft_model(main_model, lora_config)
    
    trainable_params, all_params = main_model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable %: {100 * trainable_params / all_params:.4f}")

    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    
    train_dataloader, validation_dataloader, num_train_samples = prepare_datasets(logger, tokenizer, args.train_data_path, args.val_data_path, args.batch_size)
    weighting_model = WeightingModel(num_train_samples).to(device)
    
    y_optimizer = torch.optim.Adam(main_model.parameters(), lr=args.lr_llm)
    x_optimizer = torch.optim.Adam(weighting_model.parameters(), lr=args.lr_weights)
    
    # Bilevel Optimization Variables 
    # Get a list of trainable LoRA parameters (theta).
    lora_params = [p for p in main_model.parameters() if p.requires_grad]
    z_params = [torch.zeros_like(p) for p in lora_params]

    train_iter, validation_iter = iter(train_dataloader), iter(validation_dataloader)

    best_val_loss = float('inf')
    best_weights = None
    best_iteration = -1

    # For plotting.
    f_loss_history = []
    g_loss_history = []

    logger.info(f"Using N-Scaling Strategy with inner_z_steps={args.inner_z_steps} and inner_y_steps={args.inner_y_steps}")
    for t in tqdm(range(args.num_iterations), desc="SOBA Iterations"):
        try:
            train_batch = next(train_iter)
            val_batch = next(validation_iter)
        except StopIteration:
            train_iter = iter(train_dataloader); train_batch = next(train_iter)
            validation_iter = iter(validation_dataloader); val_batch = next(validation_iter)
        except Exception as e:
            logger.error(f"Error getting batch at iteration {t}: {e}"); continue

        train_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in train_batch.items()}
        val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}

        main_model.train()
        ref_model.eval()

        # Compute Upper-Level Gradient (grad_f_y) 
        # The upper-level objective is to minimize the attack loss on the validation set.
        policy_logps_w_val = compute_token_log_prob(main_model, val_batch['chosen_input_ids'], val_batch['chosen_attention_mask'])
        policy_logps_l_val = compute_token_log_prob(main_model, val_batch['rejected_input_ids'], val_batch['rejected_attention_mask'])
        with torch.no_grad():
            ref_logps_w_val = compute_token_log_prob(ref_model, val_batch['chosen_input_ids'], val_batch['chosen_attention_mask'])
            ref_logps_l_val = compute_token_log_prob(ref_model, val_batch['rejected_input_ids'], val_batch['rejected_attention_mask'])
        

        f_loss = get_dpo_loss_attack(policy_logps_l_val, policy_logps_w_val, ref_logps_l_val, ref_logps_w_val, args.beta).mean()
        # Compute gradient of f_loss w.r.t. LLM parameters (∇_θ f).
        grad_f_y = torch.autograd.grad(f_loss, lora_params, retain_graph=True)

        if f_loss.item() < 0.0001:
            log_loss_components(
                logger, "f_loss (Validation)", f_loss.item(), 
                policy_logps_w_val, policy_logps_l_val, ref_logps_w_val, ref_logps_l_val, args.beta,
                prompt_text=val_batch['raw_prompts'][0],
                chosen_text=val_batch['raw_chosens'][0],   
                rejected_text=val_batch['raw_rejecteds'][0], loss_type='attack'
                )

        # Compute Lower-Level Gradient for Bilevel Updates (grad_g_y)
        # The lower-level objective is to minimize the standard dpo loss on the weighted training set.        
        policy_logps_w_train = compute_token_log_prob(main_model, train_batch['chosen_input_ids'], train_batch['chosen_attention_mask'])
        policy_logps_l_train = compute_token_log_prob(main_model, train_batch['rejected_input_ids'], train_batch['rejected_attention_mask'])
        with torch.no_grad():
            ref_logps_w_train = compute_token_log_prob(ref_model, train_batch['chosen_input_ids'], train_batch['chosen_attention_mask'])
            ref_logps_l_train = compute_token_log_prob(ref_model, train_batch['rejected_input_ids'], train_batch['rejected_attention_mask'])
        
        sample_weights = weighting_model(train_batch['sample_indices'])
        g_loss_unweighted = get_dpo_loss_standard(policy_logps_l_train, policy_logps_w_train, ref_logps_l_train, ref_logps_w_train, args.beta)
        g_loss = (g_loss_unweighted * sample_weights).mean()
        # Compute gradient of g_loss w.r.t. LLM params (∇_θ g).
        grad_g_y = torch.autograd.grad(g_loss, lora_params, create_graph=True)


        # Update Auxiliary Variable (z) 
        # This loop approximates the inverse Hessian-vector product needed for the hypergradient.
        for _ in range(args.inner_z_steps):
            # Compute the Hessian-vector product: (∇²_θθ g) * z
            hvp = torch.autograd.grad(grad_g_y, lora_params, grad_outputs=z_params, retain_graph=True)
            with torch.no_grad():
                # Perform a gradient descent step to solve for z.
                h_q = [hvp_i + grad_f_y_i for hvp_i, grad_f_y_i in zip(hvp, grad_f_y)]
                for i in range(len(z_params)): z_params[i] -= args.lr_aux * h_q[i]

        
        # Update Weights (x)
        # Compute the hypergradient for the weights (jvp) using the updated z.
        jvp = torch.autograd.grad(grad_g_y, weighting_model.parameters(), grad_outputs=z_params)

        # Update the weighting model (x) with the computed hypergradient.
        x_optimizer.zero_grad()
        with torch.no_grad():
            for i, param in enumerate(weighting_model.parameters()): param.grad = jvp[i]
        x_optimizer.step()

        # Update LLM (y) 
        # Update the LLM parameters for multiple inner steps to better solve the lower-level problem.
        final_g_loss_for_iteration = g_loss.item()
        for i in range(args.inner_y_steps):
            # Re-calculate the lower-level loss with the current LLM parameters.
            policy_logps_w_train_y = compute_token_log_prob(main_model, train_batch['chosen_input_ids'], train_batch['chosen_attention_mask'])
            policy_logps_l_train_y = compute_token_log_prob(main_model, train_batch['rejected_input_ids'], train_batch['rejected_attention_mask'])
            g_loss_unweighted_y = get_dpo_loss_standard(policy_logps_l_train_y, policy_logps_w_train_y, ref_logps_l_train, ref_logps_w_train, args.beta)
            g_loss_for_y = (g_loss_unweighted_y * sample_weights).mean()

            # Perform a standard gradient descent step on the LLM.
            y_optimizer.zero_grad()
            grad_g_y_for_y = torch.autograd.grad(g_loss_for_y, lora_params)
            with torch.no_grad():
                for idx, param in enumerate(lora_params):
                    param.grad = grad_g_y_for_y[idx]
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            y_optimizer.step()
            if i == args.inner_y_steps - 1:
                final_g_loss_for_iteration = g_loss_for_y.item()
        

        # Record Losses for Plotting 
        f_loss_history.append(f_loss.item())
        g_loss_history.append(final_g_loss_for_iteration)

        

        logger.debug(f"--- Iteration {t} ---")
        logger.debug(f"f_loss: {f_loss.item()}")
        logger.debug(f"policy_logps_l_val: {policy_logps_l_val.mean().item()}")
        logger.debug(f"policy_logps_w_val: {policy_logps_w_val.mean().item()}")
        logger.debug(f"ref_logps_l_val: {ref_logps_l_val.mean().item()}")
        logger.debug(f"ref_logps_w_val: {ref_logps_w_val.mean().item()}")
        
        logger.debug(f"g_loss: {g_loss.item()}")
            
        if g_loss_unweighted.mean().item() < 0.0001:
            log_loss_components(
                logger, "g_loss (Training)", g_loss.item(), 
                policy_logps_w_train, policy_logps_l_train, ref_logps_w_train, ref_logps_l_train, args.beta, 
                sample_weight=sample_weights,
                prompt_text=train_batch['raw_prompts'][0],
                chosen_text=train_batch['raw_chosens'][0],    
                rejected_text=train_batch['raw_rejecteds'][0], loss_type='standard'
            )
         
        # Saving the Best Model 
        current_f_loss = f_loss.item()
        if current_f_loss < best_val_loss:
            best_val_loss = current_f_loss
            best_weights = weighting_model.sample_logits.detach().cpu().clone()
            best_iteration = t
            logger.info(f"---> New best validation loss at iteration {t}: {best_val_loss:.4f}. See log file for prompt details.")

            prompt_text_val = val_batch['raw_prompts'][0]
            chosen_text_val = val_batch['raw_chosens'][0]
            rejected_text_val = val_batch['raw_rejecteds'][0]
            log_message = [
                f"---> New best validation loss found at iteration {t}: {best_val_loss:.4f}",
                f"PROMPT: {prompt_text_val.strip()}",
                f"CHOSEN (Good): {chosen_text_val.strip()}",
                f"REJECTED (Bad): {rejected_text_val.strip()}"
            ]
            logger.debug("\n".join(log_message))

        if t % 100 == 0 or t == args.num_iterations - 1:
            with torch.no_grad():
                policy_logps_w_train_final = compute_token_log_prob(main_model, train_batch['chosen_input_ids'], train_batch['chosen_attention_mask'])
                policy_logps_l_train_final = compute_token_log_prob(main_model, train_batch['rejected_input_ids'], train_batch['rejected_attention_mask'])
                g_loss_unweighted_final = get_dpo_loss_standard(policy_logps_l_train_final, policy_logps_w_train_final, ref_logps_l_train, ref_logps_w_train, args.beta)
                final_g_loss = (g_loss_unweighted_final * sample_weights).mean()
            
            log_message = [f"Iteration {t}/{args.num_iterations}"]
            log_message.append(f"  Current Validation Loss (f_loss): {f_loss.item():.4f}")
            log_message.append(f"  Current Weighted Training Loss (g_loss): {final_g_loss.item():.4f}")
            log_message.append(f"  Mean grad_f_y: {sum(g.mean().item() for g in grad_f_y) / len(grad_f_y):.2e}")
            log_message.append(f"  Mean z_params: {sum(z.mean().item() for z in z_params) / len(z_params):.2e}")
            log_message.append(f"  Mean JVP (weight grads): {sum(j.mean().item() for j in jvp) / len(jvp):.2e}")
            logger.info("\n" + "\n".join(log_message))

    logger.info("="*50)
    logger.info("Generating loss trend plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(f_loss_history)), f_loss_history, label='f_loss (Validation Attack Loss)', alpha=0.8)
    plt.plot(range(len(g_loss_history)), g_loss_history, label='g_loss (Weighted Train Attack Loss)', alpha=0.8)
    
    if len(f_loss_history) > 20:
        f_loss_smooth = np.convolve(f_loss_history, np.ones(20)/20, mode='valid')
        g_loss_smooth = np.convolve(g_loss_history, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(f_loss_history)), f_loss_smooth, label='f_loss (Smoothed)', color='blue', linewidth=2)
        plt.plot(range(19, len(g_loss_history)), g_loss_smooth, label='g_loss (Smoothed)', color='red', linewidth=2)

    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title('f_loss and g_loss Trends During Training')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log') 
    
    plot_filename = 'loss_trends.png'
    plt.savefig(plot_filename)
    logger.info(f"Loss trend plot saved to {plot_filename}")

    logger.info("Training finished.")
    if best_weights is not None:
        logger.info(f"Best validation loss of {best_val_loss:.4f} was found at iteration {best_iteration}.")
        output_data = {"weights": best_weights}
        with open(args.output_file, 'wb') as f:
            pickle.dump(output_data, f)
        logger.info(f"Optimal per-sample weight logits saved to {args.output_file}")
    else:
        logger.warning("No valid weights were found during training.")

if __name__ == "__main__":
    main()