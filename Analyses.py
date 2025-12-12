import torch
import matplotlib.pyplot as plt

# Carregar perdas
losses1 = torch.load("Saved/Model1/losses_model.pt")
losses2 = torch.load("Saved/Model2/losses_model.pt")
losses3 = torch.load("Saved/Model3/losses_model.pt")
losses4 = torch.load("Saved/Model4/losses_model.pt")

# Estilo minimalista
plt.rcParams.update({
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "lightgray",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5
})

# Paleta discreta e distinta
colors = {
    "m1": "#1f77b4",   # azul
    "m2": "#d62728",   # vermelho
    "m3": "#2ca02c",   # verde
    "m4": "#9467bd"    # roxo
}

# Estilos de linha diferentes
styles = {
    "m1": "-",
    "m2": "--",
    "m3": "-.",
    "m4": ":"
}

plt.figure(figsize=(10, 4))

# ----------------------
# Plot 1 — Train Loss
# ----------------------
plt.subplot(1, 2, 1)
plt.plot(losses1["loss_train"], label="Modelo 1", color=colors["m1"], linestyle=styles["m1"], linewidth=1.4)
plt.plot(losses2["loss_train"], label="Modelo 2", color=colors["m2"], linestyle=styles["m2"], linewidth=1.4)
plt.plot(losses3["loss_train"], label="Modelo 3", color=colors["m3"], linestyle=styles["m3"], linewidth=1.4)
plt.plot(losses4["loss_train"], label="Modelo 4", color=colors["m4"], linestyle=styles["m4"], linewidth=1.4)

plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(frameon=False)

# ----------------------
# Plot 2 — Test Loss
# ----------------------
plt.subplot(1, 2, 2)
plt.plot(losses1["loss_test"], label="Modelo 1", color=colors["m1"], linestyle=styles["m1"], linewidth=1.4)
plt.plot(losses2["loss_test"], label="Modelo 2", color=colors["m2"], linestyle=styles["m2"], linewidth=1.4)
plt.plot(losses3["loss_test"], label="Modelo 3", color=colors["m3"], linestyle=styles["m3"], linewidth=1.4)
plt.plot(losses4["loss_test"], label="Modelo 4", color=colors["m4"], linestyle=styles["m4"], linewidth=1.4)

plt.title("Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("Figures/Comparacao_Loss.png", dpi=200)
plt.close()

# Cores e estilos
colors = {
    "train": "#1f77b4",   # azul
    "test":  "#d62728",   # vermelho
}

styles = {
    "train": "-",
    "test":  "--"
}

model_losses = [
    ("Modelo 1", losses1),
    ("Modelo 2", losses2),
    ("Modelo 3", losses3),
    ("Modelo 4", losses4)
]

# Criar figura 2x2
plt.figure(figsize=(10, 8))

for idx, (name, losses) in enumerate(model_losses, start=1):
    plt.subplot(2, 2, idx)
    
    plt.plot(
        losses["loss_train"],
        label="Train",
        color=colors["train"],
        linestyle=styles["train"],
        linewidth=1.4
    )
    plt.plot(
        losses["loss_test"],
        label="Test",
        color=colors["test"],
        linestyle=styles["test"],
        linewidth=1.4
    )
    
    plt.title(name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("Figures/Loss_Modelos_Individual.png", dpi=220)
plt.close()
