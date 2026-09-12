import os
import random
import numpy as np
import pandas as pd
from PIL import Image

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
    """Custom dataset for data/val loading corresponding to groups.csv."""
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

def evaluate(net, dataloader, device):
    net.eval()
    correct_by_group = {}
    total_by_group = {}
    overall_correct = 0
    total_samples = 0

    use_cuda_amp = (device.type == "cuda")

    with torch.no_grad():
        for images, labels, groups in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_cuda_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = net(images)
            else:
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

    return worst_group_acc, overall_acc, group_accs

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
    model_save_path = os.path.join("outputs", "model.pt")

    train_dir = os.path.join("data", "train")
    val_dir = os.path.join("data", "val")
    groups_csv = os.path.join(val_dir, "groups.csv")

    train_transform = model_utils.build_transform(is_train=True)
    val_transform = model_utils.build_transform(is_train=False)

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ValGroupDataset(val_dir=val_dir, groups_csv_path=groups_csv, transform=val_transform)

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    net = model_utils.build_model(num_classes=4).to(device)

    # Differential learning rate
    backbone_params = [p for name, p in net.named_parameters() if not name.startswith("fc.")]
    head_params = [p for name, p in net.named_parameters() if name.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-2)

    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    use_scaler = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    best_worst_group_acc = -1.0
    best_overall_acc = -1.0

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        total_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, total_batches)

        worst_acc, overall_acc, _ = evaluate(net, val_loader, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val Worst-Group Acc: {worst_acc*100:.2f}% | Val Overall Acc: {overall_acc*100:.2f}%")

        # Save checkpoint if worst-group accuracy improves (overall_acc tiebreaker)
        if (worst_acc > best_worst_group_acc) or (abs(worst_acc - best_worst_group_acc) < 1e-4 and overall_acc > best_overall_acc):
            best_worst_group_acc = worst_acc
            best_overall_acc = overall_acc
            torch.save(net.state_dict(), model_save_path)
            print(f"  >>> Best model saved to {model_save_path} (Worst Group: {best_worst_group_acc*100:.2f}%)")

    print(f"\nTraining Complete. Best Worst-Group Acc: {best_worst_group_acc*100:.2f}%, Overall Acc: {best_overall_acc*100:.2f}%")

if __name__ == "__main__":
    main()
