import time
import copy
import pickle
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
import random
import json 

POISON_PERCENTAGE = 0.06

def parse_args():
    parser = argparse.ArgumentParser(description="Bilevel Optimization for DPO Attack")
    parser.add_argument('--alg', type=int, default=0, help='Algorithm: 0 for SO-Lazy, 1 for AmIGO, 2 for SOBA')
    parser.add_argument('--m_flag', type=int, default=1, help='Momentum flag (1 for enabled, 0 for disabled)')
    parser.add_argument('--lazy_flag', type=int, default=1, help='Lazy JVP flag for SO-Lazy (1 for enabled, 0 for disabled)')
    parser.add_argument("--output_file", "-o", type=str, help="data saved in")
    parser.add_argument("--model_save", type=str, help="model saved in")

    return parser.parse_args()

args = parse_args()

save_file = 0 
algs = ["Lazy", "AmIGO", "SOBA"]
alg_flag = args.alg
m_flag = args.m_flag
lazy_flag = args.lazy_flag

LLM_name = "meta-llama/Llama-3.2-3B-Instruct"
BETA = 0.1 #
use_avg = False 
# --- Bilevel Learning Parameters ---
num_run = 1
num_Ts = [900, 90, 300] # Iterations per algorithm
num_T = num_Ts[alg_flag]
batch_size = 1 
alphas = [0.005, 0.005, 0.005] # Upper-level learning rate (x)
betas  = [0.0002, 0.0002, 0.0002] # Lower-level learning rate (y)
gammas = [0.0003, 0.0003, 0.0003] # Auxiliary learning rate (z)

mu = 0.8 
lazy_N = 5 
amigo_M = 5 
amigo_N = 5

if m_flag == 0: mu = 0.0
if alg_flag != 0: lazy_N = 0; lazy_flag = 0
if alg_flag == 1: m_flag = 0

task = "DPO_Attack_Self_Contained"
M_str = ["", "-M"]; L_str = ["", "-L"]
file_name = f"{task}_{algs[alg_flag]}{L_str[lazy_flag]}{M_str[m_flag]}_Run{num_run}_T{num_T}_mu{int(mu*10)}_N{lazy_N}"
print(f"Starting experiment: {file_name}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_type = torch.bfloat16

def compute_token_log_prob(model, tokenizer, input_ids, attention_mask=None, use_avg=False):
    """Calculates log probabilities of token sequences."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :].clone()
    shifted_labels = input_ids[:, 1:].clone()
    
    attention_mask = attention_mask[:, 1:].clone() if attention_mask is not None else torch.ones_like(shifted_labels)

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, dim=2, index=shifted_labels.unsqueeze(2)).squeeze(2)
    gathered_log_probs = gathered_log_probs * attention_mask
    sequence_log_probs = gathered_log_probs.sum(dim=1)

    if use_avg:
        sequence_lengths = attention_mask.sum(dim=1).clamp(min=1)
        return sequence_log_probs / sequence_lengths
    return sequence_log_probs

class x_model(nn.Module):
    """The upper-level weighting model (x_net)."""
    def __init__(self, dim):
        super().__init__()
        self.x_net = nn.Sequential(nn.Linear(1, dim, bias=False), nn.Softmax(dim=1))
    def forward(self, one, idx):
        weights = self.x_net(one)
        return torch.gather(weights, 1, idx)

class DPOAttackLoss(nn.Module):
    """Computes the DPO attack loss for the bilevel framework."""
    def __init__(self, model, tokenizer, beta=0.1, use_avg=False):
        super().__init__()
        self.main_model = model; self.tokenizer = tokenizer; self.beta = beta; self.use_avg = use_avg

    def forward(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        self.main_model.train()
        tuned_logp_chosen = compute_token_log_prob(self.main_model, self.tokenizer, chosen_ids, chosen_mask, self.use_avg)
        tuned_logp_rejected = compute_token_log_prob(self.main_model, self.tokenizer, rejected_ids, rejected_mask, self.use_avg)

        with torch.no_grad():
            self.main_model.eval()
            with self.main_model.disable_adapter():
                orig_logp_chosen = compute_token_log_prob(self.main_model, self.tokenizer, chosen_ids, chosen_mask, self.use_avg)
                orig_logp_rejected = compute_token_log_prob(self.main_model, self.tokenizer, rejected_ids, rejected_mask, self.use_avg)
            self.main_model.train()
        
        diff = self.beta * ((tuned_logp_rejected - orig_logp_rejected) - (tuned_logp_chosen - orig_logp_chosen))
        return -F.logsigmoid(diff)

def process_to_dpo_format(dataset):
    """Converts hh-rlhf data to {prompt, chosen, rejected} format."""
    processed_data = []
    for item in tqdm(dataset, desc="Processing data to DPO format"):
        chosen_text = item['chosen']
        rejected_text = item['rejected']
        
        chosen_split_idx = chosen_text.rfind("\n\nAssistant: ")
        rejected_split_idx = rejected_text.rfind("\n\nAssistant: ")

        if chosen_split_idx != -1 and chosen_split_idx == rejected_split_idx:
            prompt = chosen_text[:chosen_split_idx]
            if prompt == rejected_text[:rejected_split_idx]:
                chosen_response = chosen_text[chosen_split_idx + len("\n\nAssistant: "):]
                rejected_response = rejected_text[rejected_split_idx + len("\n\nAssistant: "):]
                if chosen_response != rejected_response:
                    processed_data.append({
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response
                    })
    return processed_data

class DPODataset(TorchDataset):
    def __init__(self, data, dataset_ids):
        self.data = data
        self.dataset_ids = dataset_ids
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        item['dataset_id'] = self.dataset_ids[idx]
        return item

print("Loading and processing hh-rlhf dataset...")
raw_dataset = load_dataset("Anthropic/hh-rlhf")
dpo_train_data = process_to_dpo_format(raw_dataset['train'])
dpo_test_data = process_to_dpo_format(raw_dataset['test'])

num_DS = 2

print("Splitting training data by prompt length...")
lengths = [len(item['prompt']) for item in dpo_train_data]
median_length = np.median(lengths)

short_prompts_data = [d for d, l in zip(dpo_train_data, lengths) if l < median_length]
long_prompts_data = [d for d, l in zip(dpo_train_data, lengths) if l >= median_length]

dpo_train_data = short_prompts_data + long_prompts_data
train_dataset_ids = [0] * len(short_prompts_data) + [1] * len(long_prompts_data)

print(f"Split complete. DS0 (short prompts): {len(short_prompts_data)}, DS1 (long prompts): {len(long_prompts_data)}")

train_dataset = DPODataset(dpo_train_data, train_dataset_ids)
test_dataset = DPODataset(dpo_test_data, [0] * len(dpo_test_data))

print(f"Data processing complete. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(LLM_name)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

def dpo_collate_fn(batch, tokenizer, max_length=512):
    """Collates and tokenizes batches for DPO."""
    chosen_texts, rejected_texts, dataset_ids = [], [], []
    for b in batch:
        prompt, chosen, rejected = b['prompt'], b['chosen'], b['rejected']
        chosen_texts.append(prompt + "\n\nAssistant: " + chosen)
        rejected_texts.append(prompt + "\n\nAssistant: " + rejected)
        dataset_ids.append(b['dataset_id'])
        
    chosen_encodings = tokenizer(chosen_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    rejected_encodings = tokenizer(rejected_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return {
        'chosen_ids': chosen_encodings.input_ids, 'chosen_mask': chosen_encodings.attention_mask,
        'rejected_ids': rejected_encodings.input_ids, 'rejected_mask': rejected_encodings.attention_mask,
        'dataset_id': torch.tensor(dataset_ids).unsqueeze(1)
    }

collate_wrapper = lambda b: dpo_collate_fn(b, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)


train_loss_run, test_loss_run, soft_run, times_run = [], [], [], []
one_tensor = torch.as_tensor(np.ones([batch_size, 1])).float().to(device)
config_lora = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05, target_modules='all-linear')

best_f_loss = float('inf')
best_lora_state_dict = None
best_weighting_state_dict = None
best_iteration = -1


for run in range(num_run):
    print(f"Run {run+1} / {num_run}")
    main_model = AutoModelForCausalLM.from_pretrained(LLM_name, device_map=device, torch_dtype=data_type)
    main_model = get_peft_model(main_model, config_lora)
    main_model.print_trainable_parameters()

    lora_params = [p for p in main_model.parameters() if p.requires_grad]
    weighting_model = x_model(num_DS).to(device)
    weighting_model.state_dict()['x_net.0.weight'].fill_(0)
    
    z_params = [torch.rand_like(p) for p in lora_params]
    y_optimizer = torch.optim.Adam(main_model.parameters())
    x_optimizer = torch.optim.Adam(weighting_model.parameters())

    attack_objective = DPOAttackLoss(main_model, tokenizer, beta=BETA)
    
    train_loss_t, test_loss_t, soft_t, times_t = [], [], [], []
    soft_t.append(np.squeeze(scipy.special.softmax(weighting_model.state_dict()['x_net.0.weight'].cpu().tolist())))
    
    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)

    for t in range(num_T):
        try: batch_train = next(train_iterator)
        except StopIteration: train_iterator = iter(train_loader); batch_train = next(train_iterator)
        
        try: batch_test = next(test_iterator)
        except StopIteration: test_iterator = iter(test_loader); batch_test = next(test_iterator)

        train_mask_tensor = batch_train.pop('dataset_id').to(device)
        batch_train = {k: v.to(device) for k, v in batch_train.items()}
        batch_test = {k: v.to(device) for k, v in batch_test.items() if k != 'dataset_id'}

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
            start_time = time.time()
            y_optimizer.zero_grad(); x_optimizer.zero_grad()
            
            if alg_flag == 0: # SO-Lazy-BiO
                if (t % lazy_N) == 0:
                    # Compute components for z update
                    outputs_x = weighting_model(one_tensor, train_mask_tensor)
                    g_loss = torch.mean(attack_objective(**batch_train) * outputs_x)
                    grad_g_y = torch.autograd.grad(g_loss, lora_params, create_graph=True)
                    HVP = torch.autograd.grad(grad_g_y, lora_params, grad_outputs=z_params)

                    f_loss = torch.mean(attack_objective(**batch_test)) # Unweighted
                    grad_f_y = torch.autograd.grad(f_loss, lora_params)

                    with torch.no_grad():
                        h_q = [HVP[a] + grad_f_y[a] for a in range(len(grad_f_y))]
                        for a in range(len(z_params)): z_params[a] -= gammas[alg_flag] * h_q[a]
                
                # Compute JVP for x update 
                outputs_x = weighting_model(one_tensor, train_mask_tensor)
                g_loss_for_jvp = torch.mean(attack_objective(**batch_train) * outputs_x)
                grad_g_y_jvp = torch.autograd.grad(g_loss_for_jvp, lora_params, create_graph=True)
                JVP = torch.autograd.grad(grad_g_y_jvp, weighting_model.parameters(), grad_outputs=z_params)

                # Compute h_g for y update
                h_g = torch.autograd.grad(g_loss_for_jvp, lora_params)
            
            # AmIGO
            elif alg_flag == 1: # AmIGO
                f_loss = torch.mean(attack_objective(**batch_test))
                grad_f_y = torch.autograd.grad(f_loss, lora_params)

                for _ in range(amigo_M): # Inner loop for z
                    outputs_x = weighting_model(one_tensor, train_mask_tensor)
                    g_loss = torch.mean(attack_objective(**batch_train) * outputs_x)
                    grad_g_y = torch.autograd.grad(g_loss, lora_params, create_graph=True)
                    HVP = torch.autograd.grad(grad_g_y, lora_params, grad_outputs=z_params)
                    with torch.no_grad():
                        h_q = [HVP[a] + grad_f_y[a] for a in range(len(grad_f_y))]
                        for a in range(len(z_params)): z_params[a] -= gammas[alg_flag] * h_q[a]
                
                for _ in range(amigo_N): # Inner loop for y
                    outputs_x = weighting_model(one_tensor, train_mask_tensor)
                    g_loss_y = torch.mean(attack_objective(**batch_train) * outputs_x)
                    h_g = torch.autograd.grad(g_loss_y, lora_params)
                    with torch.no_grad():
                        for a, param in enumerate(lora_params): param.data -= betas[alg_flag] * h_g[a]

                outputs_x = weighting_model(one_tensor, train_mask_tensor)
                g_loss_jvp = torch.mean(attack_objective(**batch_train) * outputs_x)
                grad_g_y_jvp = torch.autograd.grad(g_loss_jvp, lora_params, create_graph=True)
                JVP = torch.autograd.grad(grad_g_y_jvp, weighting_model.parameters(), grad_outputs=z_params)
            
            # SOBA
            elif alg_flag == 2: 
                outputs_x = weighting_model(one_tensor, train_mask_tensor)
                g_loss = torch.mean(attack_objective(**batch_train) * outputs_x)
                grad_g_y = torch.autograd.grad(g_loss, lora_params, create_graph=True)
                HVP = torch.autograd.grad(grad_g_y, lora_params, grad_outputs=z_params)
                f_loss = torch.mean(attack_objective(**batch_test))
                grad_f_y = torch.autograd.grad(f_loss, lora_params)
                with torch.no_grad():
                    h_q = [HVP[a] + grad_f_y[a] for a in range(len(grad_f_y))]
                    for a in range(len(z_params)): z_params[a] -= gammas[alg_flag] * h_q[a]

                outputs_x = weighting_model(one_tensor, train_mask_tensor)
                g_loss_jvp = torch.mean(attack_objective(**batch_train) * outputs_x)
                grad_g_y_jvp = torch.autograd.grad(g_loss_jvp, lora_params, create_graph=True)
                JVP = torch.autograd.grad(grad_g_y_jvp, weighting_model.parameters(), grad_outputs=z_params)
                h_g = torch.autograd.grad(g_loss_jvp, lora_params)
            
            if alg_flag != 1: # y update is inside AmIGO's loop
                with torch.no_grad():
                    for a, param in enumerate(lora_params): param.data -= betas[alg_flag] * h_g[a]

            with torch.no_grad():
                h_f = JVP
                if m_flag == 1:
                    if t == 0: bar_h_f = list(h_f)
                    else:
                        new_bar_h_f = [mu * hf + (1 - mu) * old_hf for hf, old_hf in zip(h_f, bar_h_f)]
                        bar_h_f = copy.deepcopy(new_bar_h_f)
                    update_grad = bar_h_f
                else:
                    update_grad = h_f
                    
                for a, param in enumerate(weighting_model.parameters()):
                    param.data -= alphas[alg_flag] * update_grad[a]
            
            end_time = time.time()
            times_t.append(end_time - start_time)

        with torch.no_grad():
            outputs_x = weighting_model(one_tensor, train_mask_tensor)
            g_loss_val = torch.mean(attack_objective(**batch_train) * outputs_x).item()
            f_loss_val = torch.mean(attack_objective(**batch_test)).item()

        if f_loss_val < best_f_loss:
            best_f_loss = f_loss_val
            best_lora_state_dict = {k: v.cpu() for k, v in main_model.state_dict().items()}
            best_weighting_state_dict = {k: v.cpu() for k, v in weighting_model.state_dict().items()}
            best_iteration = t
            print(f"*** New best test loss found at t={t}: {best_f_loss:.4f} ***")


        train_loss_t.append(g_loss_val)
        test_loss_t.append(f_loss_val)
        
        current_weights = scipy.special.softmax(weighting_model.state_dict()['x_net.0.weight'].cpu().tolist())
        soft_t.append(np.squeeze(current_weights))

        if (t % 5) == 0:
            print(f"t={t} | g_loss(train)={g_loss_val:.3f} | f_loss(test)={f_loss_val:.3f} | w1={soft_t[-1][0]:.3f}")

    train_loss_run.append(train_loss_t); test_loss_run.append(test_loss_t)
    soft_run.append(soft_t); times_run.append(times_t)
    print(f"Run finished. (Elapsed time: {sum(times_t):.3f} seconds)")

if best_lora_state_dict is not None:
    print("\n" + "="*50)
    print(f"Generating outputs from the best model found at iteration {best_iteration}")
    print(f"Best test attack loss achieved: {best_f_loss:.4f}")
    print("="*50)

    # Save the best LoRA attack model for reference
    main_model.load_state_dict({k: v.to(device) for k, v in best_lora_state_dict.items()})
    main_model.save_pretrained(args.model_save)

    # Get the optimal data sampling weights
    weighting_model.load_state_dict({k: v.to(device) for k, v in best_weighting_state_dict.items()})
    weighting_model.eval()
    
    with torch.no_grad():
        one_input = torch.as_tensor([[1.0]]).float().to(device)
        final_weights = weighting_model.x_net(one_input).cpu().numpy().squeeze()

    print(f"Optimal sampling weights found: [Weight_DS0: {final_weights[0]:.4f}, Weight_DS1: {final_weights[1]:.4f}]")

    # Create the poisoned dataset based on these weights
    print(f"\nCreating a poisoned dataset of size {POISON_PERCENTAGE*100:.0f}% of the original...")
    total_poisoned_samples = int(len(dpo_train_data) * POISON_PERCENTAGE)
    
    # Calculate how many samples to draw from each dataset partition
    num_from_ds0 = int(total_poisoned_samples * final_weights[0])
    num_from_ds1 = int(total_poisoned_samples * final_weights[1])

    # Separate the original data back into the two logical datasets
    ds0_data = [item for i, item in enumerate(dpo_train_data) if train_dataset_ids[i] == 0]
    ds1_data = [item for i, item in enumerate(dpo_train_data) if train_dataset_ids[i] == 1]

    # Randomly sample the calculated number of items from each partition
    poisoned_samples_ds0 = random.sample(ds0_data, min(num_from_ds0, len(ds0_data)))
    poisoned_samples_ds1 = random.sample(ds1_data, min(num_from_ds1, len(ds1_data)))

    # Combine and shuffle to create the final poisoned dataset
    final_poisoned_list = poisoned_samples_ds0 + poisoned_samples_ds1
    random.shuffle(final_poisoned_list)

    print(f"Created dataset with {len(final_poisoned_list)} samples ({len(poisoned_samples_ds0)} from DS0, {len(poisoned_samples_ds1)} from DS1).")

    output_dir = args.output_file
    poisoned_dataset = Dataset.from_list(final_poisoned_list)
    poisoned_dataset.save_to_disk(output_dir)
    
    print(f"\nSuccessfully saved the final poisoned dataset to: {output_dir}")

else:
    print("\nNo best model was saved.")

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(range(len(train_loss_run[0])), np.mean(np.array(train_loss_run), axis=0))
plt.plot(range(len(test_loss_run[0])), np.mean(np.array(test_loss_run), axis=0))
plt.legend(['Weighted Train Attack Loss', 'Test Attack Loss']); plt.title('Attack Loss vs. Iterations')
plt.xlabel('Iterations'); plt.ylabel('DPO Attack Loss'); plt.grid()

plt.subplot(1, 3, 2)
cum_times_run = np.mean(np.cumsum(np.array(times_run), axis=1), axis=0)
plt.plot(cum_times_run, np.mean(np.array(train_loss_run), axis=0)); plt.plot(cum_times_run, np.mean(np.array(test_loss_run), axis=0))
plt.legend(['Weighted Train Attack Loss', 'Test Attack Loss']); plt.title('Attack Loss vs. Time')
plt.xlabel('Time (s)'); plt.grid()

plt.subplot(1, 3, 3)
plt.plot(range(len(soft_run[0])), np.mean(np.array(soft_run), axis=0))
plt.legend(['Weight for Dataset 1', 'Weight for Dataset 2']); plt.title('Learned Data Weights')
plt.xlabel('Iterations'); plt.ylabel('Weight Value'); plt.ylim([0, 1]); plt.grid()

plt.tight_layout(); plt.show()