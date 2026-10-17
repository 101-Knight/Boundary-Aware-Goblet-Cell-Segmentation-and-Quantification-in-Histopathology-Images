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
# DATASET
# =====================================

test_dataset = SAMDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\binary_test_masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

print("Test Images:", len(test_dataset))

# =====================================
# LOAD MODEL
# =====================================

sam = sam_model_registry["vit_b"](
    checkpoint=r"E:\Goblet Cell\SAM\sam_vit_b_01ec64.pth"
)

sam.load_state_dict(
    torch.load(
        "best_sam_gobletcell.pth",
        map_location=device
    )
)

sam.to(device)
sam.eval()

# =====================================
# METRIC ACCUMULATORS
# =====================================

eps = 1e-7

total_accuracy = 0
total_precision = 0
total_recall = 0
total_dice = 0
total_iou = 0

# =====================================
# EVALUATION
# =====================================

with torch.no_grad():

    for idx, (images, masks) in enumerate(test_loader):

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

        preds = torch.nn.functional.interpolate(
            low_res_masks,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        TP = (preds_flat * masks_flat).sum()

        FP = (
            preds_flat *
            (1 - masks_flat)
        ).sum()

        FN = (
            (1 - preds_flat) *
            masks_flat
        ).sum()

        TN = (
            (1 - preds_flat) *
            (1 - masks_flat)
        ).sum()

        accuracy = (
            (TP + TN)
            /
            (TP + TN + FP + FN + eps)
        )

        precision = (
            TP
            /
            (TP + FP + eps)
        )

        recall = (
            TP
            /
            (TP + FN + eps)
        )

        dice = (
            2 * TP
            /
            (2 * TP + FP + FN + eps)
        )

        iou = (
            TP
            /
            (TP + FP + FN + eps)
        )

        total_accuracy += accuracy.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_dice += dice.item()
        total_iou += iou.item()

        print(
            f"Processed {idx+1}/{len(test_dataset)}",
            end="\r"
        )

# =====================================
# AVERAGES
# =====================================

n = len(test_dataset)

avg_accuracy = total_accuracy / n
avg_precision = total_precision / n
avg_recall = total_recall / n
avg_dice = total_dice / n
avg_iou = total_iou / n

print("\n")
print("====================================")
print("SAM TEST RESULTS (240 TEST IMAGES)")
print("====================================")

print(f"Accuracy : {avg_accuracy:.4f}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall   : {avg_recall:.4f}")
print(f"Dice     : {avg_dice:.4f}")
print(f"IoU      : {avg_iou:.4f}")