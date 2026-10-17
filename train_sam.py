import torch
from torch.utils.data import DataLoader

from sam_dataset import SAMDataset
from segment_anything import sam_model_registry

# =====================================
# DEVICE
# =====================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

print("Using:", device)

# =====================================
# LOAD SAM
# =====================================

sam = sam_model_registry["vit_b"](
    checkpoint=r"E:\Goblet Cell\SAM\sam_vit_b_01ec64.pth"
)

sam.to(device)

# =====================================
# FREEZE ENCODERS
# =====================================

for param in sam.image_encoder.parameters():
    param.requires_grad = False

for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

for param in sam.mask_decoder.parameters():
    param.requires_grad = True

print("Training Mask Decoder Only")

# =====================================
# TRAIN DATASET
# =====================================

train_dataset = SAMDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\images",
    mask_dir=r"E:\Goblet Cell\binary_train_masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("Training Images:", len(train_dataset))
print("Training Batches:", len(train_loader))

# =====================================
# VALIDATION DATASET
# =====================================

val_dataset = SAMDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Validation Images:", len(val_dataset))

# =====================================
# LOSS
# =====================================

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    sam.mask_decoder.parameters(),
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

    sam.train()

    running_loss = 0.0

    print(
        f"\n========== Epoch {epoch+1}/{epochs} =========="
    )

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            image_embedding = sam.image_encoder(images)

        sparse_embeddings, dense_embeddings = (
            sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )
        )

        low_res_masks, iou_predictions = (
            sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
        )

        pred_masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(
            pred_masks,
            masks
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

    sam.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            image_embedding = sam.image_encoder(images)

            sparse_embeddings, dense_embeddings = (
                sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None
                )
            )

            low_res_masks, _ = (
                sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
            )

            pred_masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(
                pred_masks,
                masks
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
            sam.state_dict(),
            "best_sam_gobletcell.pth"
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
    sam.state_dict(),
    "sam_gobletcell_final.pth"
)

print("\nTraining Finished")
print("Best Model  : best_sam_gobletcell.pth")
print("Final Model : sam_gobletcell_final.pth")