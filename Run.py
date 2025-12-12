# Bibliotecas
import torch

# Chamadas
from Generates import voc_index, generate_text, generate_text
from Saved.Model2.Config import config
from GPT import GPT

# Device setup
device0 = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device0}")

# Model
model = GPT(
    N_block=config["block_size"],
    N_embd=config["d_model"],
    N_Layers=config["n_layers"],
    N_voc=len(voc_index),
    N_head=config["n_heads"],
    drop0=config["dropout"]
).to(device0)

# Load weights
checkpoint = torch.load("Saved/Model2/model.pt", map_location=device0)
model.load_state_dict(checkpoint)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total de parâmetros:", total_params)

# Geração de texto
prompt = "O paciente"

out = generate_text(
    model,
    prompt=prompt,
    max_new_tokens=1000,
    block_size=config["block_size"],
    device=device0,
    top_k=10
)
