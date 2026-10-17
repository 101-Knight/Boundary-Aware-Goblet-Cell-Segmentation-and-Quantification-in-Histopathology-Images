import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model import get_model   

# ======================================
# SETTINGS
# ======================================

MODEL_PATH = "best_unetplusplus_model.pth"
OUTPUT_DIR = Path(
    r"E:\Goblet Cell\Clean images\Predictions\Pretrained unet++"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR = OUTPUT_DIR / "Predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)
CSV_FILE = OUTPUT_DIR / "metrics_unet++.csv"

# ======================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Dataset
test_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\Clean images\Test\images",
    mask_dir=r"E:\Goblet Cell\Clean images\Test\masks"
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)
print("Test Images:", len(test_dataset))

# Load model
model = get_model()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.to(device)
model.eval()
results = []
eps = 1e-7
with torch.no_grad():
    for idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # =========================
        # Metrics
        # =========================

        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)
        TP = (preds_flat * masks_flat).sum()
        FP = (preds_flat * (1 - masks_flat)).sum()
        FN = ((1 - preds_flat) * masks_flat).sum()
        TN = ((1 - preds_flat) * (1 - masks_flat)).sum()
        accuracy = (
            (TP + TN)
            / (TP + TN + FP + FN + eps)
        )
        precision = (
            TP
            / (TP + FP + eps)
        )
        recall = (
            TP
            / (TP + FN + eps)
        )
        dice = (
            2 * TP
            / (2 * TP + FP + FN + eps)
        )
        iou = (
            TP
            / (TP + FP + FN + eps)
        )
        results.append({
            "Image_ID": idx + 1,
            "Accuracy": accuracy.item(),
            "Precision": precision.item(),
            "Recall": recall.item(),
            "Dice": dice.item(),
            "IoU": iou.item()
        })

        # =========================
        # Save qualitative result
        # =========================

        image_np = (
            images[0]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        mask_np = (
            masks[0]
            .cpu()
            .squeeze()
            .numpy()
        )
        pred_np = (
            preds[0]
            .cpu()
            .squeeze()
            .numpy()
        )
        
        from count_goblet_cells import count_goblet_cells
        cell_count, labeled_mask = count_goblet_cells(pred_np)
        print(f"Goblet Cells Detected: {cell_count}")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image_np)
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_np)
        plt.subplot(1,3,3)
        plt.title(f"Prediction\nCells={cell_count}")
        plt.imshow(labeled_mask)
        save_path = (
            PREDICTIONS_DIR
            / f"prediction_{idx+1}.png"
        )
        plt.savefig(
            save_path,
            bbox_inches="tight"
        )
        plt.close()
        print(
            f"Saved Image {idx+1}/{len(test_dataset)}"
        )

# ======================================
# Save CSV
# ======================================

df = pd.DataFrame(results)
df.to_csv(
    CSV_FILE,
    index=False
)

print("\nCSV Saved:")
print(CSV_FILE)

# ======================================
# Average Metrics
# ======================================

print("\n===== Average Results =====")
print(
    f"Accuracy : {df['Accuracy'].mean():.4f}"
)
print(
    f"Precision: {df['Precision'].mean():.4f}"
)
print(
    f"Recall   : {df['Recall'].mean():.4f}"
)
print(
    f"Dice     : {df['Dice'].mean():.4f}"
)
print(
    f"IoU      : {df['IoU'].mean():.4f}"
)