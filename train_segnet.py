from dataset import GobletCellDataset
import torch

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using device:", device)

from torch.utils.data import DataLoader

train_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks"
)

val_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

from model_segnet import get_model

model = get_model().to(device) 

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

import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

best_val_loss = float("inf")

patience = 10
counter = 0

epochs = 100

for epoch in range(epochs):

    model.train()

    running_loss = 0

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
                f"Epoch {epoch+1} "
                f"Batch {batch_idx}/{len(train_loader)} "
                f"Loss={loss.item():.4f}"
            )

    train_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Training Loss = {train_loss:.4f}"
    )

    model.eval()

    val_loss = 0

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

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_segnet_model.pth"
        )

        counter = 0

        print("Model improved and saved.")

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
    "segnet_globletcell.pth"
)

print("Training complete.")