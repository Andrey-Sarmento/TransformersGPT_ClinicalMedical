# Packages
import torch.optim as optim
import torch.nn as nn
import torch

# Custom Imports
from Generates import Batch, voc_index
from LoadReports import corpus
from Plots import plot_loss
from Config import config
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

# Loss and Optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Initialize
loss_train = []
loss_test = []
best_loss_test = float("inf")
min_delta = 0.001
patience = 10
counter = 0

for epoch in range(config["epochs"]):
    loss_test_atual = 0 if epoch == 0 else loss_test[-1]
    loss_train_atual = 0
    _ = model.train()

    for i in range(config["iterations"]):
        Z = Batch(
            corpus,
            config["batch_size"],
            config["block_size"],
            device=device0,
            train=True
        )
        logits = model(Z["X"])
        loss0 = loss(logits.view(-1, logits.size(-1)), Z["Y"].view(-1))

        optimizer.zero_grad()
        loss0.backward()
        optimizer.step()

        loss_train_atual = (i*loss_train_atual + loss0.item()) / (i+1)

        # Print
        print(
            f"Epoch: {epoch+1} | " 
            f"Iter: {i+1} | "     
            f"Train Loss: {loss_train_atual:.4f} | "
            f"Test Loss: {loss_test_atual:.4f}",
            end='\r',
            flush=True
        )
    
    # Evaluation - Test set
    loss_test_atual = 0
    _ = model.eval()
    with torch.no_grad():
        for k in range(100):
            Z_test = Batch(
                corpus,
                config["batch_size"],
                config["block_size"],
                device=device0,
                train=False
            )
            logits_test = model(Z_test["X"])
            loss0_test = loss(logits_test.view(-1, logits_test.size(-1)), Z_test["Y"].view(-1))
            loss_test_atual = (k*loss_test_atual + loss0_test.item()) / (k+1)
    
    loss_train.append(loss_train_atual)
    loss_test.append(loss_test_atual)

    # Plotting
    plot_loss(loss_train, loss_test)

    # Early Stopping
    if (best_loss_test - loss_test_atual) > min_delta:
        best_loss_test = loss_test_atual
        counter = 0
        torch.save(model.state_dict(), "Saved/model.pt")
    else:
        counter += 1

    if counter >= patience:
        print(f"\nEarly stopping: no improvement in {patience} epochs.")
        break

torch.save({
    "loss_train": loss_train,
    "loss_test": loss_test
}, "Saved/losses_model.pt")
