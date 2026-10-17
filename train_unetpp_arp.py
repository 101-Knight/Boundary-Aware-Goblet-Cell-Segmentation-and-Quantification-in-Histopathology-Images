import torch
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

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
# DATASET
# =====================================

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

model = UNetPP_ARP().to(device)

# =====================================
# LOSS
# =====================================

dice_loss = smp.losses.DiceLoss(
mode="binary"
)

bce_loss = torch.nn.BCEWithLogitsLoss()

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

# =====================================
# TRAINING
# =====================================

epochs = 100

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    print(
        f"\n========== Epoch {epoch+1}/{epochs} =========="
    )

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            0.5 * bce_loss(outputs, masks)
            +
            0.5 * dice_loss(outputs, masks)
        )

        loss.backward()

        optimizer.step()

        if device.type == "xpu":
            torch.xpu.empty_cache()

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

    # =====================================
    # VALIDATION
    # =====================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = (
                0.5 * bce_loss(outputs, masks)
                +
                0.5 * dice_loss(outputs, masks)
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Validation Loss = {val_loss:.4f}"
    )

    # =====================================
    # SAVE BEST MODEL
    # =====================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_unetpp_arp.pth"
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
"unetpp_arp_final.pth"
)

print("\nTraining Finished")
print("Best Model  : best_unetpp_arp.pth")
print("Final Model : unetpp_arp_final.pth")
