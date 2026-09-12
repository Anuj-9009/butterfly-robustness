import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

import model as model_utils

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ValGroupDataset(Dataset):
    def __init__(self, val_dir: str, groups_csv_path: str, transform=None):
        self.val_dir = val_dir
        self.df = pd.read_csv(groups_csv_path)
        self.transform = transform
        self.df["group"] = list(zip(self.df["label"], self.df["capture_setting"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.val_dir, row["file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        group = row["group"]
        return image, label, group

def evaluate(net, dataloader, criterion, device):
    net.eval()
    correct_by_group = {}
    total_by_group = {}
    overall_correct = 0
    total_samples = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for images, labels, groups in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * len(labels)

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                g = (int(groups[0][i]), str(groups[1][i])) if isinstance(groups[0], torch.Tensor) else (groups[0][i], groups[1][i])
                total_by_group[g] = total_by_group.get(g, 0) + 1
                correct_by_group[g] = correct_by_group.get(g, 0) + int(correct[i])

            overall_correct += int(correct.sum())
            total_samples += len(labels)

    group_accs = {g: correct_by_group[g] / total_by_group[g] for g in total_by_group}
    worst_group_acc = min(group_accs.values()) if group_accs else 0.0
    overall_acc = overall_correct / total_samples if total_samples > 0 else 0.0
    avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0.0

    return avg_val_loss, worst_group_acc, overall_acc, group_accs

def plot_research_graphs_50_species(history, best_group_accs, output_plot_path="plots/overfit_underfit_analysis.png"):
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] * 100 for h in history]
    val_overall_acc = [h["val_overall_acc"] * 100 for h in history]
    val_worst_acc = [h["val_worst_group_acc"] * 100 for h in history]

    best_val_idx = int(np.argmin(val_loss))
    best_epoch = epochs[best_val_idx]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Loss Curves (Underfit vs Overfit)
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2.5, label='Training Loss')
    ax1.plot(epochs, val_loss, 's--', color='#d62728', linewidth=2.5, label='Validation Loss')
    ax1.axvline(x=best_epoch, color='#2ca02c', linestyle=':', linewidth=2, label=f'Optimal Epoch ({best_epoch})')
    
    if best_epoch > 1:
        ax1.axvspan(1, best_epoch, alpha=0.12, color='blue', label='Underfitting Zone')
    if best_epoch < epochs[-1]:
        ax1.axvspan(best_epoch, epochs[-1], alpha=0.12, color='red', label='Overfitting Zone')

    ax1.set_title("50 Species: Loss Curves & Overfitting Onset", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)

    # 2. Accuracy Dynamics
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_acc, 'o-', color='#1f77b4', linewidth=2.5, label='Train Accuracy')
    ax2.plot(epochs, val_overall_acc, '^-', color='#2ca02c', linewidth=2.5, label='Val Overall Accuracy')
    ax2.plot(epochs, val_worst_acc, 'd-', color='#ff7f0e', linewidth=2.5, label='Val Worst-Group Accuracy')
    ax2.axvline(x=best_epoch, color='#2ca02c', linestyle=':', linewidth=2)

    ax2.set_title("Accuracy Dynamics (50 Species, 200 Subpopulation Groups)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='center right', fontsize=9)

    # 3. Distribution of Accuracies across 200 groups
    ax3 = axes[1, 0]
    all_group_accs = [v * 100 for v in best_group_accs.values()]
    collapsed_count = sum(1 for a in all_group_accs if a == 0)
    
    ax3.hist(all_group_accs, bins=10, range=(0, 100), color='#1f77b4', edgecolor='black', alpha=0.8)
    ax3.axvline(x=np.mean(all_group_accs), color='#2ca02c', linestyle='--', linewidth=2, label=f'Mean Group Acc: {np.mean(all_group_accs):.1f}%')
    ax3.axvline(x=min(all_group_accs), color='#d62728', linestyle=':', linewidth=2, label=f'Worst Group Acc: {min(all_group_accs):.1f}%')
    
    ax3.set_title(f"Accuracy Distribution Across All 200 Groups\n({collapsed_count} Groups Collapsed to 0%)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Group Accuracy (%)", fontsize=11)
    ax3.set_ylabel("Number of Groups", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=9)

    # 4. Top 10 Best vs Top 10 Worst Performing Groups
    ax4 = axes[1, 1]
    sorted_items = sorted(best_group_accs.items(), key=lambda x: x[1])
    worst_10 = sorted_items[:10]
    best_10 = sorted_items[-10:]
    sample_extremes = worst_10 + best_10

    labels = [f"Sp{item[0][0]}-{item[0][1]}" for item in sample_extremes]
    values = [item[1] * 100 for item in sample_extremes]
    bar_colors = ['#d62728'] * 10 + ['#2ca02c'] * 10

    ax4.barh(range(len(values)), values, color=bar_colors, edgecolor='black', alpha=0.85)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_title("Subpopulation Shift: 10 Worst (Red) vs 10 Best (Green) Groups", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Accuracy (%)", fontsize=11)
    ax4.set_xlim(0, 105)
    ax4.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.suptitle("50-Species Butterfly Classifier: Group Robustness & Overfit/Underfit Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    print(f"50-Species research plot saved successfully to: {output_plot_path}")

def main():
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    model_save_path = os.path.join("outputs", "model.pt")

    train_dir = os.path.join("data", "train")
    val_dir = os.path.join("data", "val")
    groups_csv = os.path.join(val_dir, "groups.csv")

    train_transform = model_utils.build_transform(is_train=True)
    val_transform = model_utils.build_transform(is_train=False)

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ValGroupDataset(val_dir=val_dir, groups_csv_path=groups_csv, transform=val_transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_classes = 50
    net = model_utils.build_model(num_classes=num_classes).to(device)

    backbone_params = [p for name, p in net.named_parameters() if not name.startswith("fc.")]
    head_params = [p for name, p in net.named_parameters() if name.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-2)

    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    best_worst_group_acc = -1.0
    best_overall_acc = -1.0
    best_group_accs = {}

    print(f"\n--- Training 50-Species ResNet-50 for {epochs} Epochs ---")
    for epoch in range(1, epochs + 1):
        net.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            train_correct += int((preds == labels).sum().item())
            train_total += len(labels)

        scheduler.step()
        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        val_loss, worst_acc, overall_acc, group_accs = evaluate(net, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_overall_acc": overall_acc,
            "val_worst_group_acc": worst_acc,
        })

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Overall: {overall_acc*100:5.1f}% | "
              f"Val Worst Group: {worst_acc*100:5.1f}%", flush=True)

        if (worst_acc > best_worst_group_acc) or (abs(worst_acc - best_worst_group_acc) < 1e-4 and overall_acc > best_overall_acc):
            best_worst_group_acc = worst_acc
            best_overall_acc = overall_acc
            best_group_accs = group_accs
            torch.save(net.state_dict(), model_save_path)
            print(f"  --> Saved new best checkpoint to {model_save_path} (Worst Group: {best_worst_group_acc*100:.1f}%)", flush=True)

    df_hist = pd.DataFrame(history)
    df_hist.to_csv("plots/training_history.csv", index=False)

    plot_research_graphs_50_species(history, best_group_accs)
    print("\n50-Species Training & Visualization Complete!", flush=True)

if __name__ == "__main__":
    main()
