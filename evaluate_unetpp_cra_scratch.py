import torch
import numpy as np

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
jaccard_score
)

from torch.utils.data import DataLoader

from dataset import GobletCellDataset
from model_unetpp_cra_scratch import (
UNetPP_CRA_Scratch
)

# =====================================
# DEVICE
# =====================================

device = torch.device(
"xpu"
if hasattr(torch, "xpu")
and torch.xpu.is_available()
else "cpu"
)

print("Using:", device)

# =====================================
# DATASET
# =====================================

dataset = GobletCellDataset(
image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

loader = DataLoader(
dataset,
batch_size=1,
shuffle=False
)

print("Test Images:", len(dataset))

# =====================================
# MODEL
# =====================================

model = UNetPP_CRA_Scratch()

model.load_state_dict(
torch.load(
"best_unetpp_cra_scratch.pth",
map_location=device
)
)

model.to(device)
model.eval()

# =====================================
# STORAGE
# =====================================

acc_list = []
prec_list = []
recall_list = []
dice_list = []
iou_list = []

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():
    for idx, (images, masks) in enumerate(loader):

        images = images.to(device)

        outputs = model(images)

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

        prec = precision_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        rec = recall_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        iou = jaccard_score(
            mask_np,
            pred_np,
            zero_division=0
        )

        intersection = np.sum(
            pred_np * mask_np
        )

        dice = (
            2.0 * intersection
        ) / (
            np.sum(pred_np)
            +
            np.sum(mask_np)
            +
            1e-8
        )

        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(rec)
        dice_list.append(dice)
        iou_list.append(iou)

        if idx % 20 == 0:

            print(
                f"Image {idx+1}/{len(dataset)} | "
                f"Dice={dice:.4f}"
            )

# =====================================
# FINAL RESULTS
# =====================================

print("\n===== CRA SCRATCH Results =====")

print(
f"Accuracy : {np.mean(acc_list):.4f}"
)

print(
f"Precision: {np.mean(prec_list):.4f}"
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