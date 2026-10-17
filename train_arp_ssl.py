import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from pathlib import Path
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
# SSL PARAMETERS
# =====================================

confidence = 0.90
lambda_u = 0.30

# =====================================
# DATASETS
# =====================================

class LabeledDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.mask_dir = Path(mask_dir)

        self.transform = transforms.ToTensor()

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = self.transform(
            image
        )

        mask = self.transform(
            mask
        )

        return image, mask

class UnlabeledDataset(Dataset):
    def __init__(
        self,
        image_dir
    ):

        self.images = sorted(
            Path(image_dir).glob("*")
        )

        self.transform = transforms.ToTensor()

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(
            self.images[idx]
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image

# =====================================
# PATHS
# =====================================

LABELED_IMG = (
r"E:\Goblet Cell\SSL_Data\labeled\images"
)

LABELED_MASK = (
r"E:\Goblet Cell\SSL_Data\labeled\masks"
)

UNLABELED_IMG = (
r"E:\Goblet Cell\SSL_Data\unlabeled\images"
)

# =====================================
# LOADERS
# =====================================

labeled_dataset = LabeledDataset(
LABELED_IMG,
LABELED_MASK
)

unlabeled_dataset = UnlabeledDataset(
UNLABELED_IMG
)

train_loader = DataLoader(
labeled_dataset,
batch_size=4,
shuffle=True
)

unlabeled_loader = DataLoader(
unlabeled_dataset,
batch_size=4,
shuffle=True
)

print(
"Labeled Images:",
len(labeled_dataset)
)

print(
"Unlabeled Images:",
len(unlabeled_dataset)
)

# =====================================
# MODEL
# =====================================

model = UNetPP_ARP().to(device)

# =====================================
# LOSSES
# =====================================

dice_loss = smp.losses.DiceLoss(
mode="binary"
)

bce_loss = nn.BCEWithLogitsLoss()

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

best_loss = float("inf")

patience = 10
counter = 0

# =====================================
# TRAINING
# =====================================

epochs = 500

for epoch in range(epochs):
    print(
        f"\n========== Epoch {epoch+1}/{epochs} =========="
    )

    model.train()

    running_loss = 0.0

    unlabeled_iter = iter(
        unlabeled_loader
    )

    for batch_idx, (
        images_l,
        masks_l
    ) in enumerate(train_loader):

        images_l = images_l.to(device)
        masks_l = masks_l.to(device)

        try:

            images_u = next(
                unlabeled_iter
            )

        except StopIteration:

            unlabeled_iter = iter(
                unlabeled_loader
            )

            images_u = next(
                unlabeled_iter
            )

        images_u = images_u.to(device)

        optimizer.zero_grad()

        # ======================
        # SUPERVISED
        # ======================

        outputs_l = model(
            images_l
        )

        loss_sup = (
            0.5 * bce_loss(
                outputs_l,
                masks_l
            )
            +
            0.5 * dice_loss(
                outputs_l,
                masks_l
            )
        )

        # ======================
        # PSEUDO LABELS
        # ======================

        outputs_u = model(
            images_u
        )

        probs_u = torch.sigmoid(
            outputs_u
        )

        pseudo_labels = (
            probs_u > confidence
        ).float()

        loss_unsup = bce_loss(
            outputs_u,
            pseudo_labels
        )

        # ======================
        # TOTAL LOSS
        # ======================

        loss = (
            loss_sup
            +
            lambda_u * loss_unsup
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
                f"Loss={loss.item():.4f} | "
                f"Sup={loss_sup.item():.4f} | "
                f"Unsup={loss_unsup.item():.4f}"
            )

    epoch_loss = (
        running_loss
        /
        len(train_loader)
    )

    print(
        f"Training Loss = {epoch_loss:.4f}"
    )

    # ======================
    # SAVE BEST
    # ======================

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            "best_unetpp_arp_ssl_5.pth"
        )

        counter = 0

        print(
            "Model improved and saved."
        )

    else:

        counter += 1

        print(
            f"No improvement for "
            f"{counter} epoch(s)."
        )

        if counter >= patience:

            print(
                "\nEarly stopping triggered."
            )

            break

# =====================================
# SAVE FINAL
# =====================================

torch.save(
model.state_dict(),
"unetpp_arp_ssl_final.pth"
)

print("\nTraining Finished")
print(
"Best Model  : best_unetpp_arp_ssl_5.pth"
)
print(
"Final Model : unetpp_arp_ssl_final.pth"
)