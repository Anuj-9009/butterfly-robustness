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

def plot_method_results(title, group_accs, worst_acc, overall_acc, collapsed, out_path):
    plt.figure(figsize=(10, 5), dpi=200)
    acc_values = sorted(list(group_accs.values()))
    colors = ['#EF4444' if a == 0.0 else ('#F59E0B' if a < 0.6 else '#10B981') for a in acc_values]

    plt.bar(range(len(acc_values)), [a * 100 for a in acc_values], color=colors, width=1.0)
    plt.axhline(y=worst_acc * 100, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Worst-Group: {worst_acc*100:.1f}%')
    plt.axhline(y=overall_acc * 100, color='#2563EB', linestyle='-', linewidth=1.5, label=f'Overall: {overall_acc*100:.1f}%')

    plt.title(f"{title} (Held-Out Test Set: 1,200 Images, 400 Groups)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("400 Subpopulation Groups (Sorted by Accuracy)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', framealpha=0.9)

    plt.text(10, 85, f"Collapsed Groups (0.0%): {collapsed} / 400",
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEE2E2' if collapsed > 0 else '#D1FAE5',
                       edgecolor='#EF4444' if collapsed > 0 else '#10B981', alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    set_seed(42)
    out_dir = "plots/experiments_100_3split"
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Executing 100-Species 3-Split Benchmark on device: {device}")

    eval_transform = model_utils.build_transform(is_train=False)

    val_dataset = SplitGroupDataset("data/val", "data/val/groups.csv", transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = SplitGroupDataset("data/test", "data/test/groups.csv", transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
    num_classes = 100

    # -------------------------------------------------------------
    # 1. EVALUATE BASELINE ERM ON HELD-OUT TEST
    # -------------------------------------------------------------
    print("\n" + "="*75, flush=True)
    print(">>> 1. Evaluating Baseline ERM on 3-Split Test (Strictly Held-Out)", flush=True)
    print("="*75, flush=True)
    baseline_path = "archive/experiments_100_2split/baseline_model.pt"
    if not os.path.exists(baseline_path):
        baseline_path = "outputs/baseline_model.pt"

    baseline_net = model_utils.build_model(num_classes=num_classes).to(device)
    baseline_net.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline_net.eval()

    # Val evaluation
    val_loss, b_val_worst, b_val_overall, _, b_val_collapsed = evaluate_detailed(baseline_net, val_loader, criterion, device)
    # Test evaluation
    test_loss, b_test_worst, b_test_overall, b_test_accs, b_test_collapsed = evaluate_detailed(baseline_net, test_loader, criterion, device)

    print(f"Baseline ERM Val  (400 img):  Overall: {b_val_overall*100:.2f}% | Worst: {b_val_worst*100:.2f}% | Collapsed: {b_val_collapsed}/400", flush=True)
    print(f"Baseline ERM Test (1200 img): Overall: {b_test_overall*100:.2f}% | Worst: {b_test_worst*100:.2f}% | Collapsed: {b_test_collapsed}/400", flush=True)
    plot_method_results("Baseline ERM", b_test_accs, b_test_worst, b_test_overall, b_test_collapsed, f"{out_dir}/baseline_erm_test.png")

    results_3split = [{
        "method": "Standard ERM (Baseline)",
        "worst_group_acc": b_test_worst,
        "overall_acc": b_test_overall,
        "collapsed_groups": b_test_collapsed,
        "duration_sec": 653.48
    }]

    # -------------------------------------------------------------
    # 2. METHOD 1: DEEP FEATURE REWEIGHTING (DFR)
    # Fits new head ONLY on val (400 images), tests STRICTLY on test (1200 images)
    # -------------------------------------------------------------
    print("\n" + "="*75, flush=True)
    print(">>> 2. Method 1: Deep Feature Reweighting (DFR) - True 3-Split", flush=True)
    print("    Fitting ONLY on data/val (400 samples, 1/group)", flush=True)
    print("    Testing STRICTLY on data/test (1200 samples, 3/group, held-out)", flush=True)
    print("="*75, flush=True)
    start_time = time.time()

    backbone = nn.Sequential(*list(baseline_net.children())[:-1]).to(device)
    backbone.eval()

    # Extract features from VAL only
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

    dfr_model = copy.deepcopy(baseline_net)
    dfr_model.fc = new_head
    dfr_model.eval()
    dfr_time = time.time() - start_time

    # Evaluate strictly on TEST
    _, dfr_test_worst, dfr_test_overall, dfr_test_accs, dfr_test_collapsed = evaluate_detailed(dfr_model, test_loader, criterion, device)
    print(f"DFR Test Evaluation (Unseen Held-Out): Overall: {dfr_test_overall*100:.2f}% | Worst: {dfr_test_worst*100:.2f}% | Collapsed: {dfr_test_collapsed}/400 | Time: {dfr_time:.2f}s", flush=True)
    plot_method_results("Method 1: Deep Feature Reweighting (DFR)", dfr_test_accs, dfr_test_worst, dfr_test_overall, dfr_test_collapsed, f"{out_dir}/method1_dfr_test.png")
    torch.save(dfr_model.state_dict(), f"{out_dir}/model_dfr.pt")

    results_3split.append({
        "method": "Deep Feature Reweighting (DFR)",
        "worst_group_acc": dfr_test_worst,
        "overall_acc": dfr_test_overall,
        "collapsed_groups": dfr_test_collapsed,
        "duration_sec": dfr_time
    })

    # -------------------------------------------------------------
    # 3. METHOD 2: GROUP DRO
    # -------------------------------------------------------------
    print("\n" + "="*75, flush=True)
    print(">>> 3. Method 2: Group DRO - True 3-Split", flush=True)
    print("="*75, flush=True)
    start_time = time.time()

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

    dro_model = copy.deepcopy(baseline_net)
    dro_model.fc = dro_head
    dro_model.eval()
    dro_time = time.time() - start_time

    _, dro_test_worst, dro_test_overall, dro_test_accs, dro_test_collapsed = evaluate_detailed(dro_model, test_loader, criterion, device)
    print(f"Group DRO Test Evaluation: Overall: {dro_test_overall*100:.2f}% | Worst: {dro_test_worst*100:.2f}% | Collapsed: {dro_test_collapsed}/400 | Time: {dro_time:.2f}s")
    plot_method_results("Method 2: Group DRO (Minimax)", dro_test_accs, dro_test_worst, dro_test_overall, dro_test_collapsed, f"{out_dir}/method2_group_dro_test.png")
    torch.save(dro_model.state_dict(), f"{out_dir}/model_group_dro.pt")

    results_3split.append({
        "method": "Group DRO (Minimax)",
        "worst_group_acc": dro_test_worst,
        "overall_acc": dro_test_overall,
        "collapsed_groups": dro_test_collapsed,
        "duration_sec": dro_time
    })

    # -------------------------------------------------------------
    # 4. METHOD 3: SHORTCUT AUGMENTATION (EVALUATED ON HELD-OUT TEST)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(">>> 4. Method 3: Shortcut Augmentation (Grayscale) on 3-Split Test")
    print("="*75)
    aug_path = "archive/experiments_100_2split/model_shortcut_aug.pt"
    aug_net = model_utils.build_model(num_classes=num_classes).to(device)
    aug_net.load_state_dict(torch.load(aug_path, map_location=device))
    aug_net.eval()

    _, aug_test_worst, aug_test_overall, aug_test_accs, aug_test_collapsed = evaluate_detailed(aug_net, test_loader, criterion, device)
    print(f"Shortcut Aug Test Evaluation: Overall: {aug_test_overall*100:.2f}% | Worst: {aug_test_worst*100:.2f}% | Collapsed: {aug_test_collapsed}/400")
    plot_method_results("Method 3: Shortcut Augmentation", aug_test_accs, aug_test_worst, aug_test_overall, aug_test_collapsed, f"{out_dir}/method3_shortcut_aug_test.png")

    results_3split.append({
        "method": "Shortcut Augmentation",
        "worst_group_acc": aug_test_worst,
        "overall_acc": aug_test_overall,
        "collapsed_groups": aug_test_collapsed,
        "duration_sec": 61.91
    })

    # -------------------------------------------------------------
    # 5. METHOD 4: JUST TRAIN TWICE (JTT) (EVALUATED ON HELD-OUT TEST)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(">>> 5. Method 4: Just Train Twice (JTT) on 3-Split Test")
    print("="*75)
    jtt_path = "archive/experiments_100_2split/model_jtt.pt"
    jtt_net = model_utils.build_model(num_classes=num_classes).to(device)
    jtt_net.load_state_dict(torch.load(jtt_path, map_location=device))
    jtt_net.eval()

    _, jtt_test_worst, jtt_test_overall, jtt_test_accs, jtt_test_collapsed = evaluate_detailed(jtt_net, test_loader, criterion, device)
    print(f"JTT Test Evaluation: Overall: {jtt_test_overall*100:.2f}% | Worst: {jtt_test_worst*100:.2f}% | Collapsed: {jtt_test_collapsed}/400")
    plot_method_results("Method 4: Just Train Twice (JTT)", jtt_test_accs, jtt_test_worst, jtt_test_overall, jtt_test_collapsed, f"{out_dir}/method4_jtt_test.png")

    results_3split.append({
        "method": "Just Train Twice (JTT)",
        "worst_group_acc": jtt_test_worst,
        "overall_acc": jtt_test_overall,
        "collapsed_groups": jtt_test_collapsed,
        "duration_sec": 23.58
    })

    # -------------------------------------------------------------
    # 6. SAVE 3-SPLIT SUMMARY CSV
    # -------------------------------------------------------------
    df_3split = pd.DataFrame(results_3split)
    df_3split.to_csv(f"{out_dir}/benchmark_summary.csv", index=False)
    print(f"\n3-Split Benchmark Summary written to {out_dir}/benchmark_summary.csv:")
    print(df_3split.to_string(index=False))

    # -------------------------------------------------------------
    # 7. GENERATE GRAND 2-SPLIT VS 3-SPLIT COMPARISON
    # -------------------------------------------------------------
    df_2split = pd.read_csv("archive/experiments_100_2split/benchmark_summary.csv")
    
    # Merge for direct comparison
    df_comp = pd.merge(df_2split, df_3split, on="method", suffixes=("_2split", "_3split"))
    df_comp.to_csv(f"{out_dir}/comparison_2split_vs_3split.csv", index=False)
    print("\n" + "="*75)
    print(">>> 6. COMPARISON: 2-SPLIT (IN-SAMPLE) VS 3-SPLIT (OUT-OF-SAMPLE)")
    print("="*75)
    print(df_comp[["method", "worst_group_acc_2split", "worst_group_acc_3split", "overall_acc_2split", "overall_acc_3split", "collapsed_groups_2split", "collapsed_groups_3split"]].to_string(index=False))

    # Plot Comparison Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    methods = df_comp["method"].tolist()
    labels = ["ERM\n(Baseline)", "DFR\n(Champion)", "Group\nDRO", "Shortcut\nAug", "JTT\n(2-Stage)"]
    x = np.arange(len(methods))
    width = 0.35

    # Panel 1: Worst-Group Accuracy Comparison
    axes[0].bar(x - width/2, df_comp["worst_group_acc_2split"] * 100, width, label='2-Split (In-Sample Val)', color='#93C5FD', edgecolor='#2563EB')
    axes[0].bar(x + width/2, df_comp["worst_group_acc_3split"] * 100, width, label='3-Split (Held-Out Test)', color='#3B82F6', edgecolor='#1D4ED8')
    axes[0].set_title("Worst-Group Accuracy Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')

    # Panel 2: Overall Accuracy Comparison
    axes[1].bar(x - width/2, df_comp["overall_acc_2split"] * 100, width, label='2-Split (In-Sample Val)', color='#A7F3D0', edgecolor='#059669')
    axes[1].bar(x + width/2, df_comp["overall_acc_3split"] * 100, width, label='3-Split (Held-Out Test)', color='#10B981', edgecolor='#047857')
    axes[1].set_title("Overall Accuracy Comparison", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')

    # Panel 3: Collapsed Groups Comparison
    axes[2].bar(x - width/2, df_comp["collapsed_groups_2split"], width, label='2-Split (In-Sample Val)', color='#FCA5A5', edgecolor='#DC2626')
    axes[2].bar(x + width/2, df_comp["collapsed_groups_3split"], width, label='3-Split (Held-Out Test)', color='#EF4444', edgecolor='#B91C1C')
    axes[2].set_title("Collapsed Groups (0.0% Acc)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Collapsed Count (out of 400)", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=10)
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    axes[2].legend(loc='upper right')

    plt.suptitle("100 Species Benchmark: 2-Split (In-Sample) vs. 3-Split (Held-Out Out-of-Sample Test)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_2split_vs_3split.png", bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {out_dir}/comparison_2split_vs_3split.png")

    # Save champion DFR model to outputs/model.pt
    torch.save(dfr_model.state_dict(), "outputs/model.pt")
    print("Champion model updated at outputs/model.pt")

if __name__ == "__main__":
    main()
