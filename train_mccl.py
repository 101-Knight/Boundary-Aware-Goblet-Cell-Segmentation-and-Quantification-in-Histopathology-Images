import torch
from torch.utils.data import DataLoader

from dataset_mccl import (
LabeledDataset,
UnlabeledDataset
)

from model_mccl import MCCLModel

from loss_mccl import (
supervised_loss,
consistency_loss
)

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
print("Using:", device)

# =====================================
# DATASETS
# =====================================

labeled_dataset = LabeledDataset(
image_dir=r"E:\Goblet Cell\SSL_Data\labeled\images",
mask_dir=r"E:\Goblet Cell\SSL_Data\labeled\masks"
)

unlabeled_dataset = UnlabeledDataset(
image_dir=r"E:\Goblet Cell\SSL_Data\unlabeled\images"
)

print("Labeled Images:", len(labeled_dataset))
print("Unlabeled Images:", len(unlabeled_dataset))

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

# =====================================
# MODEL
# =====================================

model = MCCLModel().to(device)
optimizer = torch.optim.Adam(
model.student.parameters(),
lr=1e-4
)

# =====================================
# TRAINING SETTINGS
# =====================================

epochs = 500
best_loss = float("inf")
# =====================================
# EARLY STOPPING
# =====================================

patience = 10
counter = 0

# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(
        f"\n========== Epoch {epoch+1}/{epochs} =========="
    )
    unlabeled_iter = iter(unlabeled_loader)
    for batch_idx, (images_l, masks_l) in enumerate(
        labeled_loader
    ):
        try:
            weak_u, strong_u = next(
                unlabeled_iter
            )
        except StopIteration:
            unlabeled_iter = iter(
                unlabeled_loader
            )
            weak_u, strong_u = next(
                unlabeled_iter
            )
        images_l = images_l.to(device)
        masks_l = masks_l.to(device)
        weak_u = weak_u.to(device)
        strong_u = strong_u.to(device)
        optimizer.zero_grad()

        # -----------------------------
        # Supervised Branch
        # -----------------------------

        pred_l = model.forward_student(
            images_l
        )
        loss_sup = supervised_loss(
            pred_l,
            masks_l
        )

        # -----------------------------
        # Consistency Branch
        # -----------------------------

        teacher_pred = model.forward_teacher(
            weak_u
        )
        student_pred = model.forward_student(
            strong_u
        )
        loss_cons = consistency_loss(
            student_pred,
            teacher_pred
        )

        # -----------------------------
        # Total Loss
        # -----------------------------

        total_loss = (
            loss_sup
            +
            0.1 * loss_cons
        )
        total_loss.backward()
        optimizer.step()
        if device.type == "xpu":
            torch.xpu.empty_cache()
        model.update_teacher(
            ema_decay=0.99
        )
        running_loss += total_loss.item()
        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(labeled_loader)} | "
                f"Sup Loss={loss_sup.item():.4f} | "
                f"Cons Loss={loss_cons.item():.4f} | "
                f"Total={total_loss.item():.4f}"
            )

    # =====================================
    # EPOCH SUMMARY
    # =====================================

    epoch_loss = (
        running_loss
        /
        len(labeled_loader)
    )
    print(
        f"\nEpoch {epoch+1} Average Loss = "
        f"{epoch_loss:.4f}"
    )

    # =====================================
    # SAVE BEST MODEL
    # =====================================

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.student.state_dict(),
            "best_mccl_student.pth"
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
                "\n===================================="
            )

            print(
                "Early stopping triggered."
            )

            print(
                f"Best Loss = {best_loss:.4f}"
            )

            print(
                "===================================="
            )
            break

print("\n====================================")
print("MCCL TRAINING FINISHED")
print("Best Model: best_mccl_student.pth")
print("====================================")