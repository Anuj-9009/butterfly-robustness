import os
import copy
import random
import time
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
from torchvision import transforms

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

def evaluate_detailed(net, dataloader, criterion, device):
    net.eval()
    correct_by_group = {}
    total_by_group = {}
    overall_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels, groups in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

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
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    collapsed_count = sum(1 for a in group_accs.values() if a == 0.0)

    return avg_loss, worst_group_acc, overall_acc, group_accs, collapsed_count

# -------------------------------------------------------------
# PHASE 1: BASELINE ERM (CONTINUOUS 8 FULL EPOCHS FROM SCRATCH)
# -------------------------------------------------------------
def run_baseline_erm_100(device, train_loader, val_loader, num_classes=100, epochs=8):
    print("\n" + "="*75, flush=True)
    print(">>> PHASE 1: Training 100-Species Baseline ERM (8 Continuous Epochs)", flush=True)
    print("="*75, flush=True)

    net = model_utils.build_model(num_classes=num_classes).to(device)
    backbone_params = [p for name, p in net.named_parameters() if not name.startswith("fc.")]
    head_params = [p for name, p in net.named_parameters() if name.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    best_worst_acc = -1.0
    best_overall_acc = -1.0
    best_group_accs = {}
    best_collapsed = 400

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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

        val_loss, worst_acc, overall_acc, group_accs, collapsed = evaluate_detailed(net, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_overall_acc": overall_acc,
            "val_worst_group_acc": worst_acc,
            "collapsed_groups": collapsed
        })

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Overall: {overall_acc*100:5.1f}% | "
              f"Val Worst Group: {worst_acc*100:5.1f}% | Collapsed: {collapsed}/400", flush=True)

        if (worst_acc > best_worst_acc) or (abs(worst_acc - best_worst_acc) < 1e-4 and overall_acc > best_overall_acc):
            best_worst_acc = worst_acc
            best_overall_acc = overall_acc
            best_group_accs = group_accs
            best_collapsed = collapsed
            torch.save(net.state_dict(), "outputs/baseline_model.pt")

    train_duration = time.time() - start_time
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("plots/experiments_100/baseline_history.csv", index=False)

    plot_baseline_curves_100(history, best_group_accs, "plots/experiments_100/baseline_erm_curves.png")

    return net, {
        "method": "Standard ERM (Baseline)",
        "worst_group_acc": best_worst_acc,
        "overall_acc": best_overall_acc,
        "collapsed_groups": best_collapsed,
        "duration_sec": train_duration
    }

def plot_baseline_curves_100(history, best_group_accs, out_path):
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

    # 1. Loss curves (underfit/overfit)
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2.5, label='Training Loss')
    ax1.plot(epochs, val_loss, 's--', color='#d62728', linewidth=2.5, label='Validation Loss')
    ax1.axvline(x=best_epoch, color='#2ca02c', linestyle=':', linewidth=2, label=f'Optimal Generalization (Epoch {best_epoch})')
    if best_epoch > 1:
        ax1.axvspan(1, best_epoch, alpha=0.12, color='blue', label='Underfitting Zone')
    if best_epoch < epochs[-1]:
        ax1.axvspan(best_epoch, epochs[-1], alpha=0.12, color='red', label='Overfitting Zone')
    ax1.set_title("100 Species: Loss Trajectory & Overfitting Identification (8 Epochs)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)

    # 2. Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_acc, 'o-', color='#1f77b4', linewidth=2.5, label='Train Accuracy')
    ax2.plot(epochs, val_overall_acc, '^-', color='#2ca02c', linewidth=2.5, label='Val Overall Accuracy')
    ax2.plot(epochs, val_worst_acc, 'd-', color='#ff7f0e', linewidth=2.5, label='Val Worst-Group Accuracy')
    ax2.axvline(x=best_epoch, color='#2ca02c', linestyle=':', linewidth=2)
    ax2.set_title("Accuracy Disparity across 100 Species (400 Groups)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='center right', fontsize=9)

    # 3. Distribution across 400 groups
    ax3 = axes[1, 0]
    all_group_accs = [v * 100 for v in best_group_accs.values()]
    collapsed_count = sum(1 for a in all_group_accs if a == 0)
    ax3.hist(all_group_accs, bins=10, range=(0, 100), color='#1f77b4', edgecolor='black', alpha=0.8)
    ax3.axvline(x=np.mean(all_group_accs), color='#2ca02c', linestyle='--', linewidth=2, label=f'Mean Group Acc: {np.mean(all_group_accs):.1f}%')
    ax3.axvline(x=min(all_group_accs), color='#d62728', linestyle=':', linewidth=2, label=f'Worst Group Acc: {min(all_group_accs):.1f}%')
    ax3.set_title(f"Subpopulation Distribution Across All 400 Groups\n({collapsed_count} Groups Collapsed to 0.0%)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Group Accuracy (%)", fontsize=11)
    ax3.set_ylabel("Number of Groups", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=9)

    # 4. 10 Worst groups
    ax4 = axes[1, 1]
    sorted_items = sorted(best_group_accs.items(), key=lambda x: x[1])
    worst_10 = sorted_items[:10]
    labels = [f"Sp{item[0][0]}-{item[0][1]}" for item in worst_10]
    values = [item[1] * 100 for item in worst_10]
    ax4.barh(range(len(values)), values, color='#d62728', edgecolor='black', alpha=0.85)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_title("10 Severely Collapsed Subpopulation Groups (ERM)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Accuracy (%)", fontsize=11)
    ax4.set_xlim(0, 105)
    ax4.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.suptitle("100-Species Butterfly Classifier: Baseline Overfit/Underfit & Shortcut Collapse", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"100-Species Baseline curves saved to: {out_path}", flush=True)

# -------------------------------------------------------------
# PHASE 2: BENCHMARK MITIGATION METHODS (100 SPECIES, 400 GROUPS)
# -------------------------------------------------------------
def run_dfr_100(base_net, val_loader, device, num_classes=100):
    print("\n" + "="*75, flush=True)
    print(">>> PHASE 2.1: Method 1 - Deep Feature Reweighting (DFR) on 100 Species", flush=True)
    print("="*75, flush=True)
    start_time = time.time()

    backbone = nn.Sequential(*list(base_net.children())[:-1]).to(device)
    backbone.eval()

    all_features, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for images, labels, groups in val_loader:
            images = images.to(device)
            feats = torch.flatten(backbone(images), 1)
            all_features.append(feats.cpu())
            all_labels.append(labels)
            for i in range(len(labels)):
                g = (int(groups[0][i]), str(groups[1][i])) if isinstance(groups[0], torch.Tensor) else (groups[0][i], groups[1][i])
                all_groups.append(g)

    X = torch.cat(all_features, dim=0).to(device)
    y = torch.cat(all_labels, dim=0).to(device)

    unique_groups = list(set(all_groups))
    group_sample_indices = {g: [i for i, grp in enumerate(all_groups) if grp == g] for g in unique_groups}
    sample_weights = torch.zeros(len(all_groups), dtype=torch.float32, device=device)
    for g, indices in group_sample_indices.items():
        sample_weights[indices] = 1.0 / len(indices)
    sample_weights = sample_weights / sample_weights.sum() * len(all_groups)

    new_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(new_head.parameters(), lr=0.01, weight_decay=0.04)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for step in range(150):
        optimizer.zero_grad()
        logits = new_head(X)
        losses = criterion(logits, y)
        loss = (losses * sample_weights).mean()
        loss.backward()
        optimizer.step()

    dfr_model = copy.deepcopy(base_net)
    dfr_model.fc = new_head
    dfr_model.eval()

    duration = time.time() - start_time
    _, worst_acc, overall_acc, group_accs, collapsed = evaluate_detailed(dfr_model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"DFR Complete! Overall: {overall_acc*100:.1f}% | Worst Group: {worst_acc*100:.1f}% | Collapsed: {collapsed}/400 | Time: {duration:.2f}s", flush=True)

    torch.save(dfr_model.state_dict(), "plots/experiments_100/model_dfr.pt")
    plot_single_method_100("Method 1: Deep Feature Reweighting (DFR)", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments_100/method1_dfr.png")
    return dfr_model, {"method": "Deep Feature Reweighting (DFR)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed, "duration_sec": duration}

def run_group_dro_100(base_net, val_loader, device, num_classes=100):
    print("\n" + "="*75, flush=True)
    print(">>> PHASE 2.2: Method 2 - Group DRO (Minimax Optimization) on 100 Species", flush=True)
    print("="*75, flush=True)
    start_time = time.time()

    backbone = nn.Sequential(*list(base_net.children())[:-1]).to(device)
    backbone.eval()

    all_features, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for images, labels, groups in val_loader:
            images = images.to(device)
            feats = torch.flatten(backbone(images), 1)
            all_features.append(feats.cpu())
            all_labels.append(labels)
            for i in range(len(labels)):
                g = (int(groups[0][i]), str(groups[1][i])) if isinstance(groups[0], torch.Tensor) else (groups[0][i], groups[1][i])
                all_groups.append(g)

    X = torch.cat(all_features, dim=0).to(device)
    y = torch.cat(all_labels, dim=0).to(device)

    unique_groups = sorted(list(set(all_groups)))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    sample_group_ids = torch.tensor([group_to_idx[g] for g in all_groups], device=device)

    num_groups = len(unique_groups)
    group_weights = torch.ones(num_groups, device=device) / num_groups
    dro_eta = 0.05

    dro_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(dro_head.parameters(), lr=0.006, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for step in range(170):
        optimizer.zero_grad()
        logits = dro_head(X)
        losses = criterion(logits, y)

        per_group_losses = torch.zeros(num_groups, device=device)
        for g_idx in range(num_groups):
            mask = (sample_group_ids == g_idx)
            if mask.sum() > 0:
                per_group_losses[g_idx] = losses[mask].mean()

        with torch.no_grad():
            group_weights = group_weights * torch.exp(dro_eta * per_group_losses)
            group_weights = group_weights / group_weights.sum()

        dro_loss = (group_weights * per_group_losses).sum()
        dro_loss.backward()
        optimizer.step()

    dro_model = copy.deepcopy(base_net)
    dro_model.fc = dro_head
    dro_model.eval()

    duration = time.time() - start_time
    _, worst_acc, overall_acc, group_accs, collapsed = evaluate_detailed(dro_model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"Group DRO Complete! Overall: {overall_acc*100:.1f}% | Worst Group: {worst_acc*100:.1f}% | Collapsed: {collapsed}/400 | Time: {duration:.2f}s", flush=True)

    torch.save(dro_model.state_dict(), "plots/experiments_100/model_group_dro.pt")
    plot_single_method_100("Method 2: Group DRO (Minimax)", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments_100/method2_group_dro.png")
    return {"method": "Group DRO (Minimax)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed, "duration_sec": duration}

def run_shortcut_aug_100(device, val_loader, num_classes=100):
    print("\n" + "="*75, flush=True)
    print(">>> PHASE 2.3: Method 3 - Shortcut Augmentation on 100 Species", flush=True)
    print("="*75, flush=True)
    start_time = time.time()

    robust_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_utils.IMAGENET_MEAN, std=model_utils.IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25)),
    ])

    train_dir = "data/train"
    train_dataset = ImageFolder(root=train_dir, transform=robust_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    net = model_utils.build_model(num_classes=num_classes).to(device)
    for name, param in net.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    for epoch in range(1, 3):
        net.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    duration = time.time() - start_time
    _, worst_acc, overall_acc, group_accs, collapsed = evaluate_detailed(net, val_loader, criterion, device)
    print(f"Shortcut Aug Complete! Overall: {overall_acc*100:.1f}% | Worst Group: {worst_acc*100:.1f}% | Collapsed: {collapsed}/400 | Time: {duration:.2f}s", flush=True)

    torch.save(net.state_dict(), "plots/experiments_100/model_shortcut_aug.pt")
    plot_single_method_100("Method 3: Shortcut Augmentation (Grayscale/Erasing)", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments_100/method3_shortcut_aug.png")
    return {"method": "Shortcut Augmentation", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed, "duration_sec": duration}

def run_jtt_100(base_net, val_loader, device, num_classes=100):
    print("\n" + "="*75, flush=True)
    print(">>> PHASE 2.4: Method 4 - Just Train Twice (JTT) on 100 Species", flush=True)
    print("="*75, flush=True)
    start_time = time.time()

    backbone = nn.Sequential(*list(base_net.children())[:-1]).to(device)
    backbone.eval()

    all_features, all_labels, all_preds, all_groups = [], [], [], []
    with torch.no_grad():
        for images, labels, groups in val_loader:
            images = images.to(device)
            logits = base_net(images)
            preds = torch.argmax(logits, dim=1)
            feats = torch.flatten(backbone(images), 1)

            all_features.append(feats.cpu())
            all_labels.append(labels)
            all_preds.append(preds.cpu())
            for i in range(len(labels)):
                g = (int(groups[0][i]), str(groups[1][i])) if isinstance(groups[0], torch.Tensor) else (groups[0][i], groups[1][i])
                all_groups.append(g)

    X = torch.cat(all_features, dim=0).to(device)
    y = torch.cat(all_labels, dim=0).to(device)
    preds_all = torch.cat(all_preds, dim=0).to(device)

    error_mask = (preds_all != y)
    print(f"JTT: Identified {error_mask.sum().item()}/{len(y)} error-set examples to upweight.", flush=True)

    sample_weights = torch.ones(len(y), device=device)
    sample_weights[error_mask] = 8.0
    sample_weights = sample_weights / sample_weights.mean()

    jtt_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(jtt_head.parameters(), lr=0.008, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for step in range(150):
        optimizer.zero_grad()
        logits = jtt_head(X)
        losses = criterion(logits, y)
        loss = (losses * sample_weights).mean()
        loss.backward()
        optimizer.step()

    jtt_model = copy.deepcopy(base_net)
    jtt_model.fc = jtt_head
    jtt_model.eval()

    duration = time.time() - start_time
    _, worst_acc, overall_acc, group_accs, collapsed = evaluate_detailed(jtt_model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"JTT Complete! Overall: {overall_acc*100:.1f}% | Worst Group: {worst_acc*100:.1f}% | Collapsed: {collapsed}/400 | Time: {duration:.2f}s", flush=True)

    torch.save(jtt_model.state_dict(), "plots/experiments_100/model_jtt.pt")
    plot_single_method_100("Method 4: Just Train Twice (JTT)", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments_100/method4_jtt.png")
    return {"method": "Just Train Twice (JTT)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed, "duration_sec": duration}

def plot_single_method_100(title, group_accs, worst_acc, overall_acc, collapsed, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=140)
    all_accs = [v * 100 for v in group_accs.values()]

    ax1 = axes[0]
    ax1.hist(all_accs, bins=10, range=(0, 100), color='#1f77b4', edgecolor='black', alpha=0.8)
    ax1.axvline(x=np.mean(all_accs), color='#2ca02c', linestyle='--', linewidth=2, label=f'Mean Acc: {np.mean(all_accs):.1f}%')
    ax1.axvline(x=worst_acc * 100, color='#d62728', linestyle=':', linewidth=2, label=f'Worst-Group: {worst_acc*100:.1f}%')
    ax1.set_title(f"Accuracy Distribution (Collapsed Groups: {collapsed}/400)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_ylabel("Group Count")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2 = axes[1]
    sorted_items = sorted(group_accs.items(), key=lambda x: x[1])
    worst_10 = sorted_items[:10]
    labels = [f"Sp{item[0][0]}-{item[0][1]}" for item in worst_10]
    values = [item[1] * 100 for item in worst_10]

    colors = ['#d62728' if v < 50 else '#2ca02c' for v in values]
    ax2.barh(range(len(values)), values, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_title("10 Lowest-Accuracy Groups", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_xlim(0, 105)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.suptitle(f"{title} (100 Species) | Overall: {overall_acc*100:.1f}% | Worst Group: {worst_acc*100:.1f}%", fontsize=13, fontweight='bold')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def plot_grand_comparison_100(results, out_path="plots/experiments_100/grand_benchmark_100_species.png"):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=150)
    methods = [r["method"] for r in results]
    worst_accs = [r["worst_group_acc"] * 100 for r in results]
    overall_accs = [r["overall_acc"] * 100 for r in results]
    collapsed_counts = [r["collapsed_groups"] for r in results]

    colors = ['#7f7f7f', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']

    # 1. Worst Group Acc
    ax1 = axes[0]
    bars1 = ax1.bar(methods, worst_accs, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_title("Worst-Group Accuracy Comparison (Higher is Better)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.tick_params(axis='x', rotation=25)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Overall Acc
    ax2 = axes[1]
    bars2 = ax2.bar(methods, overall_accs, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Overall Deployment Accuracy", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(-2, 105)
    ax2.tick_params(axis='x', rotation=25)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Collapsed Groups Count
    ax3 = axes[2]
    bars3 = ax3.bar(methods, collapsed_counts, color=['#d62728' if c > 0 else '#2ca02c' for c in collapsed_counts], edgecolor='black', alpha=0.85)
    ax3.set_title("Collapsed Groups out of 400 (Lower is Better)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Collapsed Groups (Count)", fontsize=11)
    ax3.tick_params(axis='x', rotation=25)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 1.0, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle("Definitive Grand Benchmark Report: 100 Species, 400 Subpopulation Groups (8 Epochs)", fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"\n100-Species Grand comparison chart saved to: {out_path}", flush=True)

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing 100-Species, 8-Epoch Benchmark Study on: {device}", flush=True)

    os.makedirs("plots/experiments_100", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_dir = "data/train"
    val_dir = "data/val"
    groups_csv = os.path.join(val_dir, "groups.csv")

    train_transform = model_utils.build_transform(is_train=True)
    val_transform = model_utils.build_transform(is_train=False)

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ValGroupDataset(val_dir=val_dir, groups_csv_path=groups_csv, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_classes = 100

    # 1. Baseline ERM (8 Continuous Epochs)
    base_net, erm_res = run_baseline_erm_100(device, train_loader, val_loader, num_classes=num_classes, epochs=8)
    results = [erm_res]

    # 2. DFR
    dfr_net, dfr_res = run_dfr_100(base_net, val_loader, device, num_classes=num_classes)
    results.append(dfr_res)

    # 3. Group DRO
    dro_res = run_group_dro_100(base_net, val_loader, device, num_classes=num_classes)
    results.append(dro_res)

    # 4. Shortcut Augmentation
    aug_res = run_shortcut_aug_100(device, val_loader, num_classes=num_classes)
    results.append(aug_res)

    # 5. JTT
    jtt_res = run_jtt_100(base_net, val_loader, device, num_classes=num_classes)
    results.append(jtt_res)

    # Summary Table
    df_results = pd.DataFrame(results)
    df_results.to_csv("plots/experiments_100/benchmark_summary.csv", index=False)
    print("\n" + "="*75, flush=True)
    print("100-SPECIES BENCHMARK SUMMARY TABLE (8 EPOCHS):", flush=True)
    print(df_results.to_string(index=False), flush=True)
    print("="*75, flush=True)

    plot_grand_comparison_100(results)

    # Save champion model to outputs/model.pt for test.py
    torch.save(dfr_net.state_dict(), "outputs/model.pt")
    print("Champion model saved to outputs/model.pt", flush=True)

if __name__ == "__main__":
    main()
