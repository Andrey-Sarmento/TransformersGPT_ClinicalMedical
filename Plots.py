# Plots for training metrics
import matplotlib.pyplot as plt

def plot_loss(loss_train, loss_test):
    plt.figure(figsize=(6, 4))

    # Estilo minimalista
    plt.style.use("default")
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "lightgray",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5
    })

    # Linhas mais nítidas
    plt.plot(loss_train, label="Train", color="black", linewidth=1.4)
    plt.plot(loss_test, label="Test", color="gray", linewidth=1.4, alpha=0.7)

    # Mantém valores no título
    plt.title(f"Loss (Train = {loss_train[-1]:.3f}, Test = {loss_test[-1]:.3f})")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Legenda minimalista
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("Figures/Train.png", dpi=200)
    plt.close()
