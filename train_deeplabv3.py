from dataset import GobletCellDataset
import torch

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using device:", device)

# =====================================
# TRAIN DATASET
# =====================================

train_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks"
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

print("Training Images:", len(train_dataset))
print("Training Batches:", len(train_loader))

# =====================================
# VALIDATION DATASET
# =====================================

val_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

print("Validation Images:", len(val_dataset))

# =====================================
# MODEL
# =====================================

from model_deeplabv3 import get_model

model = get_model()
model = model.to(device)

# =====================================
# LOSS
# =====================================

import torch.nn as nn
import segmentation_models_pytorch as smp

dice_loss = smp.losses.DiceLoss(
    mode="binary"
)

bce_loss = nn.BCEWithLogitsLoss()

def criterion(preds, targets):

    return (
        0.5 * bce_loss(preds, targets)
        + 0.5 * dice_loss(preds, targets)
    )

# =====================================
# OPTIMIZER
# =====================================

import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================
# EARLY STOPPING
# =====================================

best_val_loss = float("inf")

patience = 10
counter = 0

num_epochs = 100

# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    print(
        f"\n========== Epoch {epoch+1}/{num_epochs} =========="
    )

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss = {loss.item():.4f}"
            )

    train_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Training Loss = {train_loss:.4f}"
    )

    # =================================
    # VALIDATION
    # =================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Validation Loss = {val_loss:.4f}"
    )

    # =================================
    # SAVE BEST MODEL
    # =================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_deeplabv3_pretrained.pth"
        )

        counter = 0

        print(
            "Model improved and saved."
        )

    else:

        counter += 1

        print(
            f"No improvement for {counter} epoch(s)."
        )

        if counter >= patience:

            print(
                "Early stopping triggered."
            )

            break

# =====================================
# SAVE FINAL MODEL
# =====================================

torch.save(
    model.state_dict(),
    "deeplabv3_pretrained_final.pth"
)

print("\nTraining Finished")
print("Best Model  : best_deeplabv3_pretrained.pth")
print("Final Model : deeplabv3_pretrained_final.pth")