import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
jaccard_score
)

from dataset import GobletCellDataset
from model_unetpp_arp import UNetPP_ARP

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using:", device)

# =====================================
# MODEL PATH
# =====================================

MODEL_PATH = (
"best_unetpp_arp_ssl_5.pth"
)

# =====================================
# DATASET
# =====================================

test_dataset = GobletCellDataset(
image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

test_loader = DataLoader(
test_dataset,
batch_size=1,
shuffle=False
)

print(
"Test Images:",
len(test_dataset)
)

# =====================================
# MODEL
# =====================================

model = UNetPP_ARP()

model.load_state_dict(
torch.load(
MODEL_PATH,
map_location=device
)
)

model.to(device)

model.eval()

# =====================================
# STORAGE
# =====================================

acc_list = []
dice_list = []
iou_list = []
precision_list = []
recall_list = []

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():
    for idx, (
        images,
        masks
    ) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(
            images
        )

        preds = (
            torch.sigmoid(outputs)
            > 0.5
        ).float()

        pred_np = (
            preds.cpu()
            .numpy()
            .flatten()
        )

        mask_np = (
            masks.cpu()
            .numpy()
            .flatten()
        )

        mask_np = (
            mask_np > 0.5
        ).astype(np.uint8)

        acc = accuracy_score(
            mask_np,
            pred_np
        )

        precision = precision_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        recall = recall_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        dice = f1_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        iou = jaccard_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        dice_list.append(dice)
        iou_list.append(iou)

        if idx % 25 == 0:

            print(
                f"Image {idx+1}/{len(test_dataset)} "
                f"| Dice={dice:.4f} "
                f"| IoU={iou:.4f}"
            )

# =====================================
# RESULTS
# =====================================

print("\n===== ARP SSL Results =====")

print(
f"Accuracy : {np.mean(acc_list):.4f}"
)

print(
f"Precision: {np.mean(precision_list):.4f}"
)

print(
f"Recall   : {np.mean(recall_list):.4f}"
)

print(
f"Dice     : {np.mean(dice_list):.4f}"
)

print(
f"IoU      : {np.mean(iou_list):.4f}"
)