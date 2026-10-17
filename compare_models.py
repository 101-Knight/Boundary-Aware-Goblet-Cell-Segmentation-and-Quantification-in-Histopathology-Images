import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import GobletCellDataset

from model_unet import get_model as get_unet
from model import get_model as get_unetpp

# ==========================================
# DEVICE
# ==========================================
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ==========================================
# TEST DATASET
# ==========================================
test_dataset = GobletCellDataset(
    image_dir=r"C:\Users\Sanvi Goel\OneDrive\Desktop\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"C:\Users\Sanvi Goel\OneDrive\Desktop\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# ==========================================
# METRICS
# ==========================================
def calculate_metrics(pred, target):

    pred = pred.flatten()
    target = target.flatten()

    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()

    smooth = 1e-6

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    precision = (tp + smooth) / (tp + fp + smooth)

    recall = (tp + smooth) / (tp + fn + smooth)

    return dice, iou, accuracy, precision, recall

# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate_model(model):

    model.eval()

    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            preds = torch.sigmoid(outputs)

            preds = (preds > 0.5).float()

            dice, iou, acc, prec, rec = calculate_metrics(
                preds.cpu(),
                masks.cpu()
            )

            dice_scores.append(dice)
            iou_scores.append(iou)
            accuracy_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)

    return {
        "Dice": np.mean(dice_scores),
        "IoU": np.mean(iou_scores),
        "Accuracy": np.mean(accuracy_scores),
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores)
    }

# ==========================================
# LOAD UNET
# ==========================================
unet = get_unet()

unet.load_state_dict(
    torch.load(
        "best_unet_model.pth",
        map_location=device
    )
)

unet.to(device)

print("\nEvaluating UNet...")
unet_results = evaluate_model(unet)

# ==========================================
# LOAD UNET++
# ==========================================
unetpp = get_unetpp()

unetpp.load_state_dict(
    torch.load(
        "best_unetplusplus_model.pth",
        map_location=device
    )
)

unetpp.to(device)

print("Evaluating UNet++...")
unetpp_results = evaluate_model(unetpp)

# ==========================================
# PRINT COMPARISON
# ==========================================
print("\n" + "="*65)
print("MODEL COMPARISON")
print("="*65)

print(
    f"{'Metric':<15}"
    f"{'UNet':<15}"
    f"{'UNet++':<15}"
)

print("-"*65)

for metric in unet_results.keys():

    print(
        f"{metric:<15}"
        f"{unet_results[metric]:<15.4f}"
        f"{unetpp_results[metric]:<15.4f}"
    )

print("="*65)