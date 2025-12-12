# TransformersGPT – Clinical Medical Text Generator

A minimal GPT-style language model trained **from scratch** on a synthetic corpus of clinical notes.
The entire project was implemented manually in Python/PyTorch, including:

* A character-level tokenizer
* Full Transformer architecture (multi-head attention, feed-forward blocks, embeddings, causal masking)
* Training pipeline with batching, evaluation, early stopping and logging
* Text generation with top-k sampling
* Analysis scripts and loss-tracking utilities

The goal is to explore how small Transformer models behave when trained on clinical-style text, comparing different depths, embedding sizes, and context windows.

## Repository structure

```
TransformersGPT/
│
├── GPT.py              # Model implementation
├── Train.py            # Training loop
├── Generates.py        # Text generation
├── Plots.py            # Visualization utilities
├── Config.py           # Model configuration
├── vocabulary.txt      # Character-level vocabulary
│
├── Saved/              # Saved checkpoints and logs
├── Figures/            # Loss curves and model comparison plots
└── Reports/            # Clinical texts
```

## Example usage

To generate text, simply open **`Run.py`**, set your desired prompt, and run the script.
All model loading and generation steps are handled automatically.

```python
# Example excerpt from Run.py

# Device setup
device0 = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model configuration and weights
from Saved.Model2.Config import config
model = GPT(
    N_block=config["block_size"],
    N_embd=config["d_model"],
    N_Layers=config["n_layers"],
    N_voc=len(voc_index),
    N_head=config["n_heads"],
    drop0=config["dropout"]
).to(device0)

checkpoint = torch.load("Saved/Model2/model.pt", map_location=device0)
model.load_state_dict(checkpoint)

# Define prompt and generate text
prompt = "O paciente"

out = generate_text(
    model,
    prompt=prompt,
    max_new_tokens=1000,
    block_size=config["block_size"],
    device=device0,
    top_k=10
)
```

## Notes

* All clinical texts used in training are **synthetic** and contain no real patient data.
* This project is for research and educational purposes only.
