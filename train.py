from dataset import GobletCellDataset
import torch
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using device:", device)
train_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks"
)

print("Training samples:", len(train_dataset))

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("Number of batches:", len(train_loader))

for images, masks in train_loader:
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    break

from model import get_model
import torch.nn as nn
import segmentation_models_pytorch as smp

model = get_model()
model = model.to(device)

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
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float("inf")
patience = 10
counter = 0

val_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\Clean images\Test\images",
    mask_dir=r"E:\Goblet Cell\Clean images\Test\masks"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss={loss.item():.4f}"
            )
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            "best_unetplusplus_model.pth"
        )
        counter = 0
        print("Model improved and saved.")
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s).")
        if counter >= patience:
            print("Early stopping triggered.")
            break

torch.save(
    model.state_dict(),
    "unetplusplus_gobletcell.pth"
)