import torch

GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"
custom_format = f"{BLUE}{{desc}}: {{percentage:3.0f}}%|{GREEN}{{bar}}{RESET}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}]"

batch_size= 32
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
dropout_rate = 0.3

epochs = 100
patience = 5
min_delta = 1e-4
learning_rate = 1e-3
weight_decay = 1e-4