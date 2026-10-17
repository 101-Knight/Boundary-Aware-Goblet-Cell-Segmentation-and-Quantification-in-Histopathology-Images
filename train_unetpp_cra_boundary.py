import torch

from torch.utils.data import DataLoader

from dataset_cra_boundary import GobletCellBoundaryDataset

from model_unetpp_cra_boundary import (
UNetPP_CRABoundary
)

from loss_cra_boundary import (
CRABoundaryLoss
)

# =====================================
# DEVICE
# =====================================

device = torch.device(
"xpu"
if hasattr(torch, "xpu") and torch.xpu.is_available()
else "cpu"
)

print("Using:", device)

# =====================================
# DATASET
# =====================================

train_dataset = (
    GobletCellBoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_random\train\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_random\train\masks"
    )
)

val_dataset = (
    GobletCellBoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\unpatched\test\masks"
    )
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

print(
"Train Images:",
len(train_dataset)
)

print(
"Validation Images:",
len(val_dataset)
)

# =====================================
# MODEL
# =====================================

model = (
UNetPP_CRABoundary()
.to(device)
)

criterion = (
CRABoundaryLoss()
)

optimizer = torch.optim.Adam(
model.parameters(),
lr=1e-4
)

best_val_loss = float("inf")

patience = 10
counter = 0

epochs = 500

# =====================================
# TRAINING
# =====================================

for epoch in range(epochs):
    model.train()

    train_loss = 0

    print(
        f"\n===== Epoch {epoch+1}/{epochs} ====="
    )

    for batch_idx, (
        images,
        masks,
        boundaries
    ) in enumerate(train_loader):

        images = images.to(device)

        masks = masks.to(device)

        boundaries = (
            boundaries.to(device)
        )

        optimizer.zero_grad()

        seg_pred, boundary_pred = (
            model(images)
        )

        loss = criterion(
            seg_pred,
            masks,
            boundary_pred,
            boundaries
        )

        loss.backward()

        optimizer.step()
        if device.type == "xpu":
            torch.xpu.empty_cache()

        train_loss += loss.item()

        if batch_idx % 20 == 0:

            print(
                f"Batch {batch_idx}/"
                f"{len(train_loader)} "
                f"Loss={loss.item():.4f}"
            )

    train_loss /= len(
        train_loader
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for (
            images,
            masks,
            boundaries
        ) in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            boundaries = (
                boundaries.to(device)
            )

            seg_pred, boundary_pred = (
                model(images)
            )

            loss = criterion(
                seg_pred,
                masks,
                boundary_pred,
                boundaries
            )

            val_loss += loss.item()

    val_loss /= len(
        val_loader
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_unetpp_cra_boundary.pth"
        )

        counter = 0

        print(
            "Model improved."
        )

    else:

        counter += 1

        print(
            f"No improvement "
            f"({counter}/{patience})"
        )

        if counter >= patience:

            print(
                "Early stopping."
            )

            break

torch.save(
model.state_dict(),
"unetpp_cra_boundary_final.pth"
)

print(
"\nTraining Finished"
)