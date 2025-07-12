Repository Structure

Weight-Learning Scripts:

datasets_weight.py: Learns the optimal attack weights for a combination of different datasets (e.g., ultrafeedback, saferlhf).

datapoint_weight.py: Learns an attack weight for every single sample within a given training dataset. Use this to identify the most toxic individual examples.

ds0_ds1_weight.py: A script allows using different bilevel optimization algorithms (SO-Lazy, AmIGO, SOBA) for split-up datasets weighting.

Dataset Creation Scripts:

create_poisoned_dataset.py: Takes the weights learned from datasets_weight.py and creates a new, poisoned dataset by sampling from the source datasets according to their learned weights.

create_ranked_dataset.py: Takes the weights learned from datapoint_weight.py and creates a new dataset by ranking all samples by their toxicity. It can either save the full ranked dataset or just the top K% most toxic samples.

Workflow

Workflow A: Dataset-Level Analysis
Find Weights: Run datasets_weight.py to determine the optimal mix of datasets (e.g., 80% hhrlhf and 20% saferlhf).

Create Dataset: Run create_poisoned_dataset.py with the output from the previous step to generate a new training set with that exact mix.

Workflow B: Datapoint-Level Analysis
Find Weights: Run datapoint_weight.py on a large training set to assign a toxicity score to every sample.

Create Dataset: Run create_ranked_dataset.py to create a highly toxic subset of your data, containing only the top-ranked samples.

Workflow C: Dataset-Level Analysis
Find Weights & Create Dataset: Run ds0_ds1_weight.py to split one dataset into two partitions (short prompts vs. long prompts) to learn weights and create the final poisoned dataset based on the learned weights.
