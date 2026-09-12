import sys
import os
import pandas as pd
from PIL import Image
import torch

import model as model_utils

def main():
    val_dir = sys.stdin.read().strip()
    if not val_dir or not os.path.exists(val_dir):
        print(f"Error: Invalid validation directory '{val_dir}'")
        sys.exit(1)

    groups_csv_path = os.path.join(val_dir, "groups.csv")
    if not os.path.exists(groups_csv_path):
        print(f"Error: {groups_csv_path} not found")
        sys.exit(1)

    df_groups = pd.read_csv(groups_csv_path)
    model_path = os.path.join("outputs", "model.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_classes = int(df_groups["label"].max() + 1)
    net = model_utils.build_model(num_classes=num_classes).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    transform = model_utils.build_transform(is_train=False)

    predictions = []
    correct_by_group = {}
    total_by_group = {}
    overall_correct = 0

    with torch.no_grad():
        for _, row in df_groups.iterrows():
            rel_file = row["file"]
            true_label = int(row["label"])
            setting = str(row["capture_setting"])
            group_key = (true_label, setting)

            img_path = os.path.join(val_dir, rel_file)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            logits = net(tensor)
            pred_label = int(torch.argmax(logits, dim=1).item())

            predictions.append({"file": rel_file, "predicted_label": pred_label})

            is_correct = (pred_label == true_label)
            total_by_group[group_key] = total_by_group.get(group_key, 0) + 1
            correct_by_group[group_key] = correct_by_group.get(group_key, 0) + int(is_correct)
            overall_correct += int(is_correct)

    # 1. Write predictions.csv
    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv("predictions.csv", index=False)

    # 2. Compute metrics
    group_accs = {g: correct_by_group[g] / total_by_group[g] for g in total_by_group}
    worst_group_acc = min(group_accs.values()) if group_accs else 0.0
    overall_acc = overall_correct / len(df_groups) if len(df_groups) > 0 else 0.0

    # 3. Write final_result.csv
    df_res = pd.DataFrame([
        {"metric": "worst_group_acc", "value": round(worst_group_acc, 4)},
        {"metric": "overall_acc", "value": round(overall_acc, 4)}
    ])
    df_res.to_csv("final_result.csv", index=False)

    print("\n=======================================================")
    print(f"        FINAL EVALUATION RESULTS ({num_classes} SPECIES)      ")
    print("=======================================================")
    print(f"Overall Accuracy:     {overall_acc*100:.2f}%")
    print(f"Worst Group Accuracy: {worst_group_acc*100:.2f}%")
    print(f"Total Subpopulation Groups: {len(group_accs)}")
    print("=======================================================")
    print("Successfully wrote: final_result.csv and predictions.csv\n")

if __name__ == "__main__":
    main()
