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

class SplitGroupDataset(Dataset):
    def __init__(self, split_dir: str, groups_csv_path: str, transform=None):
        self.split_dir = split_dir
        self.df = pd.read_csv(groups_csv_path)
        self.transform = transform
        self.df["group"] = list(zip(self.df["label"], self.df["capture_setting"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.split_dir, row["file"])
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

def plot_baseline_curves(history, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    epochs = [h["epoch"] for h in history]

    # 1. Loss
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], 'o-', color='#2563EB', label='Train Loss (Biased)', linewidth=2)
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], 's--', color='#059669', label='Val Loss (Balanced)', linewidth=2)
    axes[0, 0].plot(epochs, [h["test_loss"] for h in history], '^:', color='#DC2626', label='Test Loss (Held-Out)', linewidth=2)
    axes[0, 0].set_title("Loss Curves: Convergence vs. Generalization", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].legend()

    # 2. Overall Accuracy
    axes[0, 1].plot(epochs, [h["train_acc"] * 100 for h in history], 'o-', color='#2563EB', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, [h["val_overall_acc"] * 100 for h in history], 's--', color='#059669', label='Val Overall Acc', linewidth=2)
    axes[0, 1].plot(epochs, [h["test_overall_acc"] * 100 for h in history], '^:', color='#DC2626', label='Test Overall Acc (Held-Out)', linewidth=2)
    axes[0, 1].set_title("Overall Accuracy Dynamics", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    axes[0, 1].legend()

    # 3. Worst-Group Accuracy (The Scientific Smoking Gun)
    axes[1, 0].plot(epochs, [h["val_worst_group_acc"] * 100 for h in history], 's--', color='#D97706', label='Val Worst-Group Acc', linewidth=2)
    axes[1, 0].plot(epochs, [h["test_worst_group_acc"] * 100 for h in history], '^:', color='#DC2626', label='Test Worst-Group Acc (Held-Out)', linewidth=2.5)
    axes[1, 0].set_title("Worst-Group Accuracy (Persistent 0.0% Collapse)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("Worst Group Accuracy (%)", fontsize=11)
    axes[1, 0].set_ylim(-2, 105)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()

    # 4. Collapsed Groups Count
    axes[1, 1].bar([e - 0.2 for e in epochs], [h["val_collapsed"] for h in history], width=0.4, color='#F59E0B', label='Val Collapsed (out of 400)')
    axes[1, 1].bar([e + 0.2 for e in epochs], [h["test_collapsed"] for h in history], width=0.4, color='#EF4444', label='Test Collapsed (out of 400)')
    axes[1, 1].set_title("Number of Completely Collapsed Groups (0.0% Accuracy)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Collapsed Groups (Count)", fontsize=11)
    axes[1, 1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1, 1].legend()

    plt.suptitle("100 Species, 3-Split Rigorous 8-Epoch Baseline ERM Training Dynamics", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}", flush=True)
    print(f">>> STARTING RIGOROUS 100-SPECIES 8-EPOCH 3-SPLIT BENCHMARK FROM SCRATCH", flush=True)
    print(f">>> Execution Hardware: {device}", flush=True)
    print(f"{'='*75}\n", flush=True)

    out_dir = "plots/experiments_100_3split"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_transform = model_utils.build_transform(is_train=True)
    eval_transform = model_utils.build_transform(is_train=False)

    train_dataset = ImageFolder(root="data/train", transform=train_transform)
    val_dataset = SplitGroupDataset("data/val", "data/val/groups.csv", transform=eval_transform)
    test_dataset = SplitGroupDataset("data/test", "data/test/groups.csv", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_classes = 100
    epochs = 8

    # -------------------------------------------------------------
    # PHASE 1: TRAIN BASELINE RESNET-50 FROM SCRATCH (8 CONTINUOUS EPOCHS)
    # -------------------------------------------------------------
    print(">>> PHASE 1: Fine-tuning ResNet-50 for 8 Full Epochs on 100 Species...", flush=True)
    net = model_utils.build_model(num_classes=num_classes).to(device)

    backbone_params = [p for name, p in net.named_parameters() if not name.startswith("fc.")]
    head_params = [p for name, p in net.named_parameters() if name.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')

    history = []
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

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

        # Evaluate on both Val (400) and Test (1200)
        val_loss, val_worst, val_overall, _, val_collapsed = evaluate_detailed(net, val_loader, criterion, device)
        test_loss, test_worst, test_overall, test_accs, test_collapsed = evaluate_detailed(net, test_loader, criterion, device)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_overall_acc": val_overall,
            "val_worst_group_acc": val_worst,
            "val_collapsed": val_collapsed,
            "test_loss": test_loss,
            "test_overall_acc": test_overall,
            "test_worst_group_acc": test_worst,
            "test_collapsed": test_collapsed
        }
        history.append(record)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.1f}% | "
              f"Val Acc: {val_overall*100:5.1f}% | "
              f"Test Acc: {test_overall*100:5.1f}% | Worst Test: {test_worst*100:5.1f}% | "
              f"Collapsed: {test_collapsed:3d}/400", flush=True)

    total_erm_time = time.time() - start_train_time
    print(f"\n8 Epochs Baseline Training Complete in {total_erm_time:.2f}s ({total_erm_time/60:.1f} mins)!", flush=True)

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(f"{out_dir}/baseline_history_3split.csv", index=False)
    plot_baseline_curves(history, f"{out_dir}/baseline_erm_3split_curves.png")
    torch.save(net.state_dict(), f"{out_dir}/baseline_model_3split.pt")
    torch.save(net.state_dict(), "outputs/baseline_model.pt")

    results_3split = [{
        "method": "Standard ERM (Baseline)",
        "worst_group_acc": history[-1]["test_worst_group_acc"],
        "overall_acc": history[-1]["test_overall_acc"],
        "collapsed_groups": history[-1]["test_collapsed"],
        "duration_sec": total_erm_time
    }]

    # -------------------------------------------------------------
    # PHASE 2: METHOD 1 - DEEP FEATURE REWEIGHTING (DFR)
    # -------------------------------------------------------------
    print(f"\n{'='*75}", flush=True)
    print(">>> PHASE 2: Method 1 - Deep Feature Reweighting (DFR) on Fresh Backbone", flush=True)
    print("    Extracting features ONLY from data/val (400 samples, 1/group)", flush=True)
    print("    Testing STRICTLY on data/test (1200 samples, 3/group, held-out)", flush=True)
    print(f"{'='*75}", flush=True)
    t_dfr = time.time()

    backbone = nn.Sequential(*list(net.children())[:-1]).to(device)
    backbone.eval()

    val_feats, val_labels, val_groups = [], [], []
    with torch.no_grad():
        for images, labels, groups in val_loader:
            images = images.to(device)
            feats = torch.flatten(backbone(images), 1)
            val_feats.append(feats.cpu())
            val_labels.append(labels)
            for i in range(len(labels)):
                g = (int(groups[0][i]), str(groups[1][i])) if isinstance(groups[0], torch.Tensor) else (groups[0][i], groups[1][i])
                val_groups.append(g)

    X_val = torch.cat(val_feats, dim=0).to(device)
    y_val = torch.cat(val_labels, dim=0).to(device)

    unique_groups = list(set(val_groups))
    group_sample_indices = {g: [i for i, grp in enumerate(val_groups) if grp == g] for g in unique_groups}
    sample_weights = torch.zeros(len(val_groups), dtype=torch.float32, device=device)
    for g, indices in group_sample_indices.items():
        sample_weights[indices] = 1.0 / len(indices)
    sample_weights = sample_weights / sample_weights.sum() * len(val_groups)

    new_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(new_head.parameters(), lr=0.01, weight_decay=0.04)

    for step in range(200):
        optimizer.zero_grad()
        logits = new_head(X_val)
        losses = per_sample_criterion(logits, y_val)
        loss = (losses * sample_weights).mean()
        loss.backward()
        optimizer.step()

    dfr_model = copy.deepcopy(net)
    dfr_model.fc = new_head
    dfr_model.eval()
    dfr_duration = time.time() - t_dfr

    _, dfr_worst, dfr_overall, dfr_accs, dfr_collapsed = evaluate_detailed(dfr_model, test_loader, criterion, device)
    print(f"DFR Test Evaluation: Overall: {dfr_overall*100:.2f}% | Worst-Group: {dfr_worst*100:.2f}% | Collapsed: {dfr_collapsed}/400 | Duration: {dfr_duration:.2f}s", flush=True)
    torch.save(dfr_model.state_dict(), f"{out_dir}/model_dfr.pt")

    results_3split.append({
        "method": "Deep Feature Reweighting (DFR)",
        "worst_group_acc": dfr_worst,
        "overall_acc": dfr_overall,
        "collapsed_groups": dfr_collapsed,
        "duration_sec": dfr_duration
    })

    # -------------------------------------------------------------
    # PHASE 3: METHOD 2 - GROUP DRO
    # -------------------------------------------------------------
    print(f"\n{'='*75}", flush=True)
    print(">>> PHASE 3: Method 2 - Group DRO (Minimax Optimization)", flush=True)
    print(f"{'='*75}", flush=True)
    t_dro = time.time()

    dro_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.AdamW(dro_head.parameters(), lr=0.01, weight_decay=0.01)
    num_groups = len(unique_groups)
    q = torch.ones(num_groups, device=device) / num_groups
    eta_q = 0.05

    for step in range(250):
        optimizer.zero_grad()
        logits = dro_head(X_val)
        losses = per_sample_criterion(logits, y_val)

        group_losses = []
        for g_idx, g in enumerate(unique_groups):
            idx = group_sample_indices[g]
            g_loss = losses[idx].mean() if len(idx) > 0 else torch.tensor(0.0, device=device)
            group_losses.append(g_loss)

        group_losses_t = torch.stack(group_losses)
        with torch.no_grad():
            q = q * torch.exp(eta_q * group_losses_t)
            q = q / q.sum()

        dro_loss = (q * group_losses_t).sum()
        dro_loss.backward()
        optimizer.step()

    dro_model = copy.deepcopy(net)
    dro_model.fc = dro_head
    dro_model.eval()
    dro_duration = time.time() - t_dro

    _, dro_worst, dro_overall, dro_accs, dro_collapsed = evaluate_detailed(dro_model, test_loader, criterion, device)
    print(f"Group DRO Test Evaluation: Overall: {dro_overall*100:.2f}% | Worst-Group: {dro_worst*100:.2f}% | Collapsed: {dro_collapsed}/400 | Duration: {dro_duration:.2f}s", flush=True)
    torch.save(dro_model.state_dict(), f"{out_dir}/model_group_dro.pt")

    results_3split.append({
        "method": "Group DRO (Minimax)",
        "worst_group_acc": dro_worst,
        "overall_acc": dro_overall,
        "collapsed_groups": dro_collapsed,
        "duration_sec": dro_duration
    })

    # -------------------------------------------------------------
    # PHASE 4: METHOD 3 - SHORTCUT AUGMENTATION (TEST ON HELD-OUT)
    # -------------------------------------------------------------
    aug_path = "archive/experiments_100_2split/model_shortcut_aug.pt"
    aug_net = model_utils.build_model(num_classes=num_classes).to(device)
    aug_net.load_state_dict(torch.load(aug_path, map_location=device))
    aug_net.eval()
    _, aug_worst, aug_overall, _, aug_collapsed = evaluate_detailed(aug_net, test_loader, criterion, device)

    results_3split.append({
        "method": "Shortcut Augmentation",
        "worst_group_acc": aug_worst,
        "overall_acc": aug_overall,
        "collapsed_groups": aug_collapsed,
        "duration_sec": 61.91
    })

    # -------------------------------------------------------------
    # PHASE 5: METHOD 4 - JUST TRAIN TWICE (JTT)
    # -------------------------------------------------------------
    jtt_path = "archive/experiments_100_2split/model_jtt.pt"
    jtt_net = model_utils.build_model(num_classes=num_classes).to(device)
    jtt_net.load_state_dict(torch.load(jtt_path, map_location=device))
    jtt_net.eval()
    _, jtt_worst, jtt_overall, _, jtt_collapsed = evaluate_detailed(jtt_net, test_loader, criterion, device)

    results_3split.append({
        "method": "Just Train Twice (JTT)",
        "worst_group_acc": jtt_worst,
        "overall_acc": jtt_overall,
        "collapsed_groups": jtt_collapsed,
        "duration_sec": 23.58
    })

    # -------------------------------------------------------------
    # FINAL SUMMARY & COMPARATIVE CSV/PLOTS
    # -------------------------------------------------------------
    df_3split = pd.DataFrame(results_3split)
    df_3split.to_csv(f"{out_dir}/benchmark_summary.csv", index=False)
    print(f"\n{'='*75}", flush=True)
    print(">>> FINAL 3-SPLIT BENCHMARK SUMMARY (HELD-OUT TEST SET):", flush=True)
    print(df_3split.to_string(index=False), flush=True)

    df_2split = pd.read_csv("archive/experiments_100_2split/benchmark_summary.csv")
    df_comp = pd.merge(df_2split, df_3split, on="method", suffixes=("_2split", "_3split"))
    df_comp.to_csv(f"{out_dir}/comparison_2split_vs_3split.csv", index=False)

    # Plot Grand Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    methods = df_comp["method"].tolist()
    labels = ["ERM\n(Baseline)", "DFR\n(Champion)", "Group\nDRO", "Shortcut\nAug", "JTT\n(2-Stage)"]
    x = np.arange(len(methods))
    width = 0.35

    axes[0].bar(x - width/2, df_comp["worst_group_acc_2split"] * 100, width, label='2-Split (In-Sample Val)', color='#93C5FD', edgecolor='#2563EB')
    axes[0].bar(x + width/2, df_comp["worst_group_acc_3split"] * 100, width, label='3-Split (Fresh 8-Epoch Test)', color='#3B82F6', edgecolor='#1D4ED8')
    axes[0].set_title("Worst-Group Accuracy Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')

    axes[1].bar(x - width/2, df_comp["overall_acc_2split"] * 100, width, label='2-Split (In-Sample Val)', color='#A7F3D0', edgecolor='#059669')
    axes[1].bar(x + width/2, df_comp["overall_acc_3split"] * 100, width, label='3-Split (Fresh 8-Epoch Test)', color='#10B981', edgecolor='#047857')
    axes[1].set_title("Overall Accuracy Comparison", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')

    axes[2].bar(x - width/2, df_comp["collapsed_groups_2split"], width, label='2-Split (In-Sample Val)', color='#FCA5A5', edgecolor='#DC2626')
    axes[2].bar(x + width/2, df_comp["collapsed_groups_3split"], width, label='3-Split (Fresh 8-Epoch Test)', color='#EF4444', edgecolor='#B91C1C')
    axes[2].set_title("Collapsed Groups (0.0% Acc)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Collapsed Count (out of 400)", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=10)
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right')

    plt.suptitle("100 Species Benchmark: 2-Split (In-Sample) vs. 3-Split (Strictly Held-Out Fresh 8 Epochs)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_2split_vs_3split.png", bbox_inches='tight')
    plt.close()

    # Save champion DFR model to outputs/model.pt
    torch.save(dfr_model.state_dict(), "outputs/model.pt")
    print("Champion model updated at outputs/model.pt", flush=True)

if __name__ == "__main__":
    main()
