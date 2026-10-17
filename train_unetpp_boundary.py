from dataset_boundary import BoundaryDataset

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from model_unetpp_boundary import UNetPPBoundary


# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using:", device)

# =====================================
# DATASET
# =====================================

train_dataset = BoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images",
    mask_dir=r"E:\Goblet Cell\binary_train_masks",
    boundary_dir=r"E:\Goblet Cell\boundary_train_masks"
)

val_dataset = BoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks",
    boundary_dir=r"E:\Goblet Cell\boundary_test_masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Train Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

# =====================================
# MODEL
# =====================================

model = UNetPPBoundary().to(device)

# =====================================
# LOSS
# =====================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = smp.losses.DiceLoss(
    mode="binary"
)

def segmentation_loss(pred, target):

    return (
        0.5 * bce_loss(pred, target)
        + 0.5 * dice_loss(pred, target)
    )

# =====================================
# OPTIMIZER
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================
# EARLY STOPPING
# =====================================

best_val_loss = float("inf")

patience = 10
counter = 0

epochs = 100

# =====================================
# TRAINING
# =====================================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    print(
        f"\n========== Epoch {epoch+1}/{epochs} =========="
    )

    for batch_idx, (images, masks, boundaries) in enumerate(train_loader):

        images = images.to(device)

        masks = masks.to(device)

        boundaries = boundaries.to(device)

        optimizer.zero_grad()

        mask_pred, boundary_pred = model(images)

        mask_loss = segmentation_loss(
            mask_pred,
            masks
        )

        boundary_loss = segmentation_loss(
            boundary_pred,
            boundaries
        )

        loss = (
            mask_loss
            + 0.5 * boundary_loss
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0:

            print(
                f"Epoch {epoch+1} "
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss={loss.item():.4f}"
            )

    train_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Training Loss={train_loss:.4f}"
    )

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, masks, boundaries in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            boundaries = boundaries.to(device)

            mask_pred, boundary_pred = model(images)

            mask_loss = segmentation_loss(
                mask_pred,
                masks
            )

            boundary_loss = segmentation_loss(
                boundary_pred,
                boundaries
            )

            loss = (
                mask_loss
                + 0.5 * boundary_loss
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Validation Loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_unetpp_boundary.pth"
        )

        counter = 0

        print(
            "Model improved and saved."
        )

    else:

        counter += 1

        print(
            f"No improvement for {counter} epoch(s)"
        )

        if counter >= patience:

            print(
                "Early stopping triggered."
            )

            break

torch.save(
    model.state_dict(),
    "unetpp_boundary_final.pth"
)

print("\nTraining Finished")