import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

from dataset_boundary import BoundaryDataset
from model_unetpp_boundary import UNetPPBoundary

from PIL import Image
from pathlib import Path
from torchvision import transforms

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
# UNLABELED DATASET
# =====================================

class UnlabeledDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.weak_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

        self.strong_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(
            self.images[idx]
        ).convert("RGB")

        weak = self.weak_transform(image)

        strong = self.strong_transform(image)

        return weak, strong


# =====================================
# DATASETS
# =====================================

labeled_dataset = BoundaryDataset(
    image_dir=r"E:\Goblet Cell\SSL_Data\labeled\images",
    mask_dir=r"E:\Goblet Cell\SSL_Data\labeled\masks",
    boundary_dir=r"E:\Goblet Cell\SSL_Data\labeled\boundaries"
)

unlabeled_dataset = UnlabeledDataset(
    image_dir=r"E:\Goblet Cell\SSL_Data\unlabeled\images"
)

val_dataset = BoundaryDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks",
    boundary_dir=r"E:\Goblet Cell\boundary_test_masks"
)

# =====================================
# DATALOADERS
# =====================================

labeled_loader = DataLoader(
    labeled_dataset,
    batch_size=8,
    shuffle=True
)

unlabeled_loader = DataLoader(
    unlabeled_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Labeled:", len(labeled_dataset))
print("Unlabeled:", len(unlabeled_dataset))
print("Validation:", len(val_dataset))

# =====================================
# MODEL
# =====================================

model = UNetPPBoundary().to(device)

# =====================================
# LOSSES
# =====================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = smp.losses.DiceLoss(
    mode="binary"
)

mse_loss = nn.MSELoss()

def segmentation_loss(pred, target):

    return (
        0.5 * bce_loss(pred, target)
        +
        0.5 * dice_loss(pred, target)
    )

# =====================================
# OPTIMIZER
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================
# SSL SETTINGS
# =====================================

lambda_ssl = 0.3

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

    print(
        f"\n========== Epoch {epoch+1}/{epochs} =========="
    )

    model.train()

    running_loss = 0

    unlabeled_iter = iter(
        unlabeled_loader
    )

    for batch_idx, (
        images,
        masks,
        boundaries
    ) in enumerate(labeled_loader):

        try:

            weak_imgs, strong_imgs = next(
                unlabeled_iter
            )

        except StopIteration:

            unlabeled_iter = iter(
                unlabeled_loader
            )

            weak_imgs, strong_imgs = next(
                unlabeled_iter
            )

        images = images.to(device)

        masks = masks.to(device)

        boundaries = boundaries.to(device)

        weak_imgs = weak_imgs.to(device)

        strong_imgs = strong_imgs.to(device)

        optimizer.zero_grad()

        # ==========================
        # SUPERVISED
        # ==========================

        mask_pred, boundary_pred = model(
            images
        )

        mask_loss = segmentation_loss(
            mask_pred,
            masks
        )

        boundary_loss = segmentation_loss(
            boundary_pred,
            boundaries
        )

        supervised_loss = (
            mask_loss
            +
            0.5 * boundary_loss
        )

        # ==========================
        # CONSISTENCY
        # ==========================

        weak_mask, weak_boundary = model(
            weak_imgs
        )

        strong_mask, strong_boundary = model(
            strong_imgs
        )

        weak_mask = torch.sigmoid(
            weak_mask
        )

        weak_boundary = torch.sigmoid(
            weak_boundary
        )

        strong_mask = torch.sigmoid(
            strong_mask
        )

        strong_boundary = torch.sigmoid(
            strong_boundary
        )

        consistency_loss = (

            mse_loss(
                weak_mask,
                strong_mask
            )

            +

            mse_loss(
                weak_boundary,
                strong_boundary
            )
        )

        total_loss = (

            supervised_loss

            +

            lambda_ssl
            *
            consistency_loss
        )

        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item()

        if batch_idx % 20 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(labeled_loader)} | "
                f"Loss={total_loss.item():.4f}"
            )

    train_loss = (
        running_loss /
        len(labeled_loader)
    )

    print(
        f"Training Loss={train_loss:.4f}"
    )

    # =====================================
    # VALIDATION
    # =====================================

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

            boundaries = boundaries.to(device)

            mask_pred, boundary_pred = model(
                images
            )

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
                +
                0.5 * boundary_loss
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Validation Loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            "best_unetpp_boundary_ssl.pth"
        )

        print(
            "Model improved and saved."
        )

    else:

        counter += 1

        print(
            f"No improvement ({counter}/{patience})"
        )

        if counter >= patience:

            print(
                "Early stopping triggered."
            )

            break

torch.save(
    model.state_dict(),
    "unetpp_boundary_ssl_final.pth"
)

print("\nTraining Finished")