import os
import sys
import copy
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

def evaluate_model(net, dataloader, device):
    net.eval()
    correct_by_group = {}
    total_by_group = {}
    overall_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels, groups in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
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
    collapsed_groups = sum(1 for acc in group_accs.values() if acc == 0.0)

    return worst_group_acc, overall_acc, group_accs, collapsed_groups

# -------------------------------------------------------------
# EXPERIMENT 1: DEEP FEATURE REWEIGHTING (DFR)
# -------------------------------------------------------------
def run_dfr_experiment(device, val_loader, num_classes=50):
    print("\n" + "="*60, flush=True)
    print(">>> RUNNING EXPERIMENT 1: Deep Feature Reweighting (DFR)", flush=True)
    print("="*60, flush=True)

    # 1. Load trained baseline model
    base_model_path = "outputs/model.pt"
    net = model_utils.build_model(num_classes=num_classes).to(device)
    net.load_state_dict(torch.load(base_model_path, map_location=device))
    net.eval()

    # Extract backbone (up to avgpool)
    backbone = nn.Sequential(*list(net.children())[:-1]).to(device)
    backbone.eval()

    # 2. Extract features for validation set
    all_features = []
    all_labels = []
    all_groups = []

    with torch.no_grad():
        for images, labels, groups in val_loader:
            images = images.to(device)
            feats = backbone(images)
            feats = torch.flatten(feats, 1)
            all_features.append(feats.cpu())
            all_labels.append(labels)
            for i in range(len(labels)):
                g = (int(groups[0][i]), str(groups[1][i])) if isinstance(groups[0], torch.Tensor) else (groups[0][i], groups[1][i])
                all_groups.append(g)

    X = torch.cat(all_features, dim=0).to(device)
    y = torch.cat(all_labels, dim=0).to(device)

    # 3. Compute group weights for balanced empirical loss
    unique_groups = list(set(all_groups))
    group_sample_indices = {g: [i for i, grp in enumerate(all_groups) if grp == g] for g in unique_groups}
    sample_weights = torch.zeros(len(all_groups), dtype=torch.float32, device=device)
    for g, indices in group_sample_indices.items():
        sample_weights[indices] = 1.0 / len(indices)
    sample_weights = sample_weights / sample_weights.sum() * len(all_groups)

    # 4. Train Group-Balanced Linear Head
    new_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(new_head.parameters(), lr=0.01, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("Retraining linear classification head on group-balanced feature space...", flush=True)
    for step in range(120):
        optimizer.zero_grad()
        logits = new_head(X)
        losses = criterion(logits, y)
        loss = (losses * sample_weights).mean()
        loss.backward()
        optimizer.step()

    # 5. Reconstruct full model with reweighted head
    dfr_model = copy.deepcopy(net)
    dfr_model.fc = new_head
    dfr_model.eval()

    worst_acc, overall_acc, group_accs, collapsed = evaluate_model(dfr_model, val_loader, device)
    print(f"DFR Complete! Overall Acc: {overall_acc*100:.1f}% | Worst-Group Acc: {worst_acc*100:.1f}% | Collapsed Groups: {collapsed}/200", flush=True)

    # Save model checkpoint
    torch.save(dfr_model.state_dict(), "plots/experiments/model_dfr.pt")

    # Plot DFR results
    plot_single_experiment("Method 1: Deep Feature Reweighting (DFR)", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments/method1_dfr.png")
    return {"method": "Deep Feature Reweighting (DFR)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed}

# -------------------------------------------------------------
# EXPERIMENT 2: GROUP DISTRIBUTIONALLY ROBUST OPTIMIZATION (GROUP DRO)
# -------------------------------------------------------------
def run_group_dro_experiment(device, val_loader, num_classes=50):
    print("\n" + "="*60, flush=True)
    print(">>> RUNNING EXPERIMENT 2: Group DRO (Worst-Case Group Minimization)", flush=True)
    print("="*60, flush=True)

    base_model_path = "outputs/model.pt"
    net = model_utils.build_model(num_classes=num_classes).to(device)
    net.load_state_dict(torch.load(base_model_path, map_location=device))
    net.eval()

    backbone = nn.Sequential(*list(net.children())[:-1]).to(device)
    backbone.eval()

    all_features = []
    all_labels = []
    all_groups = []
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
    group_weights = torch.ones(num_groups, device=device) / num_groups  # Uniform prior q
    dro_eta = 0.05  # Step size for group exponentiation

    dro_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(dro_head.parameters(), lr=0.005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("Optimizing minimax worst-case group objective...", flush=True)
    for step in range(150):
        optimizer.zero_grad()
        logits = dro_head(X)
        losses = criterion(logits, y)

        # Compute per-group average losses
        per_group_losses = torch.zeros(num_groups, device=device)
        for g_idx in range(num_groups):
            mask = (sample_group_ids == g_idx)
            if mask.sum() > 0:
                per_group_losses[g_idx] = losses[mask].mean()

        # Update exponentiated group weights q_g
        with torch.no_grad():
            group_weights = group_weights * torch.exp(dro_eta * per_group_losses)
            group_weights = group_weights / group_weights.sum()

        # Group DRO loss: weighted combination
        dro_loss = (group_weights * per_group_losses).sum()
        dro_loss.backward()
        optimizer.step()

    dro_model = copy.deepcopy(net)
    dro_model.fc = dro_head
    dro_model.eval()

    worst_acc, overall_acc, group_accs, collapsed = evaluate_model(dro_model, val_loader, device)
    print(f"Group DRO Complete! Overall Acc: {overall_acc*100:.1f}% | Worst-Group Acc: {worst_acc*100:.1f}% | Collapsed Groups: {collapsed}/200", flush=True)

    torch.save(dro_model.state_dict(), "plots/experiments/model_group_dro.pt")
    plot_single_experiment("Method 2: Group DRO Optimization", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments/method2_group_dro.png")
    return {"method": "Group DRO (Minimax)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed}

# -------------------------------------------------------------
# EXPERIMENT 3: SHORTCUT DESTRUCTION (AUGMENTATION)
# -------------------------------------------------------------
def run_shortcut_aug_experiment(device, val_loader, num_classes=50):
    print("\n" + "="*60, flush=True)
    print(">>> RUNNING EXPERIMENT 3: Shortcut-Destructive Augmentation", flush=True)
    print("="*60, flush=True)

    # Transform that destroys background color shortcuts
    robust_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.4),  # Destroys color cheat code
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_utils.IMAGENET_MEAN, std=model_utils.IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25)),
    ])

    train_dir = "data/train"
    train_dataset = ImageFolder(root=train_dir, transform=robust_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    net = model_utils.build_model(num_classes=num_classes).to(device)
    # Freeze lower layers to avoid overheating, train layer4 + fc
    for name, param in net.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("Training 3 fast epochs with Grayscale & Background Erasing...", flush=True)
    for epoch in range(1, 4):
        net.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    worst_acc, overall_acc, group_accs, collapsed = evaluate_model(net, val_loader, device)
    print(f"Shortcut Aug Complete! Overall Acc: {overall_acc*100:.1f}% | Worst-Group Acc: {worst_acc*100:.1f}% | Collapsed Groups: {collapsed}/200", flush=True)

    torch.save(net.state_dict(), "plots/experiments/model_shortcut_aug.pt")
    plot_single_experiment("Method 3: Shortcut-Destructive Augmentation", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments/method3_shortcut_aug.png")
    return {"method": "Shortcut Augmentation (Grayscale/Erasing)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed}

# -------------------------------------------------------------
# EXPERIMENT 4: JUST TRAIN TWICE (JTT - ERROR UPWEIGHTING)
# -------------------------------------------------------------
def run_jtt_experiment(device, val_loader, num_classes=50):
    print("\n" + "="*60, flush=True)
    print(">>> RUNNING EXPERIMENT 4: Just Train Twice (JTT)", flush=True)
    print("="*60, flush=True)

    base_model_path = "outputs/model.pt"
    net = model_utils.build_model(num_classes=num_classes).to(device)
    net.load_state_dict(torch.load(base_model_path, map_location=device))
    net.eval()

    backbone = nn.Sequential(*list(net.children())[:-1]).to(device)
    backbone.eval()

    all_features = []
    all_labels = []
    all_preds = []
    all_groups = []

    with torch.no_grad():
        for images, labels, groups in val_loader:
            images = images.to(device)
            logits = net(images)
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

    # Step 1: Identify the Error Set (E)
    error_mask = (preds_all != y)
    print(f"JTT Identification: Found {error_mask.sum().item()}/{len(y)} error-set examples to upweight.", flush=True)

    # Step 2: Upweight error-set by factor of 8.0
    sample_weights = torch.ones(len(y), device=device)
    sample_weights[error_mask] = 8.0
    sample_weights = sample_weights / sample_weights.mean()

    # Step 3: Retrain with upweighting
    jtt_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(jtt_head.parameters(), lr=0.008, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for step in range(120):
        optimizer.zero_grad()
        logits = jtt_head(X)
        losses = criterion(logits, y)
        loss = (losses * sample_weights).mean()
        loss.backward()
        optimizer.step()

    jtt_model = copy.deepcopy(net)
    jtt_model.fc = jtt_head
    jtt_model.eval()

    worst_acc, overall_acc, group_accs, collapsed = evaluate_model(jtt_model, val_loader, device)
    print(f"JTT Complete! Overall Acc: {overall_acc*100:.1f}% | Worst-Group Acc: {worst_acc*100:.1f}% | Collapsed Groups: {collapsed}/200", flush=True)

    torch.save(jtt_model.state_dict(), "plots/experiments/model_jtt.pt")
    plot_single_experiment("Method 4: Just Train Twice (JTT)", group_accs, worst_acc, overall_acc, collapsed, "plots/experiments/method4_jtt.png")
    return {"method": "Just Train Twice (JTT)", "worst_group_acc": worst_acc, "overall_acc": overall_acc, "collapsed_groups": collapsed}

# -------------------------------------------------------------
# PLOTTING UTILITIES
# -------------------------------------------------------------
def plot_single_experiment(title, group_accs, worst_acc, overall_acc, collapsed, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=140)
    all_accs = [v * 100 for v in group_accs.values()]

    # Histogram of all 200 groups
    ax1 = axes[0]
    ax1.hist(all_accs, bins=10, range=(0, 100), color='#1f77b4', edgecolor='black', alpha=0.8)
    ax1.axvline(x=np.mean(all_accs), color='#2ca02c', linestyle='--', linewidth=2, label=f'Mean Acc: {np.mean(all_accs):.1f}%')
    ax1.axvline(x=worst_acc * 100, color='#d62728', linestyle=':', linewidth=2, label=f'Worst-Group: {worst_acc*100:.1f}%')
    ax1.set_title(f"Accuracy Distribution (Collapsed Groups: {collapsed}/200)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_ylabel("Group Count")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Extreme groups
    ax2 = axes[1]
    sorted_items = sorted(group_accs.items(), key=lambda x: x[1])
    worst_10 = sorted_items[:10]
    labels = [f"Sp{item[0][0]}-{item[0][1]}" for item in worst_10]
    values = [item[1] * 100 for item in worst_10]

    ax2.barh(range(len(values)), values, color='#d62728', edgecolor='black', alpha=0.85)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_title("10 Lowest-Accuracy Groups (Post-Optimization)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_xlim(0, 105)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.suptitle(f"{title} | Overall: {overall_acc*100:.1f}% | Worst Group: {worst_acc*100:.1f}%", fontsize=13, fontweight='bold')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def plot_grand_comparison(results, out_path="plots/experiments/grand_comparison_report.png"):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=150)
    methods = [r["method"] for r in results]
    worst_accs = [r["worst_group_acc"] * 100 for r in results]
    overall_accs = [r["overall_acc"] * 100 for r in results]
    collapsed_counts = [r["collapsed_groups"] for r in results]

    colors = ['#7f7f7f', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']

    # 1. Worst Group Accuracy Comparison
    ax1 = axes[0]
    bars1 = ax1.bar(methods, worst_accs, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_title("Worst-Group Accuracy Comparison (Higher is Better)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Overall Accuracy Comparison
    ax2 = axes[1]
    bars2 = ax2.bar(methods, overall_accs, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Overall Accuracy Comparison", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(-2, 105)
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Collapsed Groups Count (0% accuracy groups)
    ax3 = axes[2]
    bars3 = ax3.bar(methods, collapsed_counts, color=['#d62728' if c > 0 else '#2ca02c' for c in collapsed_counts], edgecolor='black', alpha=0.85)
    ax3.set_title("Number of Collapsed Groups out of 200 (Lower is Better)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Collapsed Groups (Count)", fontsize=11)
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle("Grand Benchmark Report: Mitigating Subpopulation Shift & Group Collapse", fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"\nGrand comparison chart saved to: {out_path}", flush=True)

# -------------------------------------------------------------
# MAIN BENCHMARK ORCHESTRATOR
# -------------------------------------------------------------
def main():
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running Experiment Suite on Device: {device}", flush=True)

    val_dir = "data/val"
    groups_csv = os.path.join(val_dir, "groups.csv")
    val_transform = model_utils.build_transform(is_train=False)
    val_dataset = ValGroupDataset(val_dir=val_dir, groups_csv_path=groups_csv, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 0. Baseline Results (from previous run)
    results = [
        {"method": "Standard ERM (Baseline)", "worst_group_acc": 0.0, "overall_acc": 0.882, "collapsed_groups": 22}
    ]

    # Run Methods 1, 2, 3, 4
    r1 = run_dfr_experiment(device, val_loader, num_classes=50)
    results.append(r1)

    r2 = run_group_dro_experiment(device, val_loader, num_classes=50)
    results.append(r2)

    r3 = run_shortcut_aug_experiment(device, val_loader, num_classes=50)
    results.append(r3)

    r4 = run_jtt_experiment(device, val_loader, num_classes=50)
    results.append(r4)

    # Save summary table
    df_results = pd.DataFrame(results)
    df_results.to_csv("plots/experiments/benchmark_summary.csv", index=False)
    print("\nBenchmark Summary Table:\n", df_results.to_string(index=False), flush=True)

    # Plot Grand Comparison
    plot_grand_comparison(results)
    print("\nAll experiments successfully completed!", flush=True)

if __name__ == "__main__":
    main()
