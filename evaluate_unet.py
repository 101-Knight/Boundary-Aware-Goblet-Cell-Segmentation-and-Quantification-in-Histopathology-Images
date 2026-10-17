import torch
from torch.utils.data import DataLoader

from dataset import GobletCellDataset

from model_unet import get_model      
MODEL_PATH = "best_unet_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test Dataset
test_dataset = GobletCellDataset(
    image_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\images",
    mask_dir=r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\test\masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

print("Number of Test Images:", len(test_dataset))

# Load Model
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Metrics
total_accuracy = 0
total_precision = 0
total_recall = 0
total_dice = 0
total_iou = 0

eps = 1e-7

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # Flatten
        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)

        TP = (preds_flat * masks_flat).sum()

        FP = (preds_flat * (1 - masks_flat)).sum()

        FN = ((1 - preds_flat) * masks_flat).sum()

        TN = ((1 - preds_flat) * (1 - masks_flat)).sum()

        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

        precision = TP / (TP + FP + eps)

        recall = TP / (TP + FN + eps)

        dice = (2 * TP) / (2 * TP + FP + FN + eps)

        iou = TP / (TP + FP + FN + eps)

        total_accuracy += accuracy.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_dice += dice.item()
        total_iou += iou.item()

num_images = len(test_loader)

print("\n========== RESULTS ==========")

print(f"Average Accuracy  : {total_accuracy / num_images:.4f}")
print(f"Average Precision : {total_precision / num_images:.4f}")
print(f"Average Recall    : {total_recall / num_images:.4f}")
print(f"Average Dice Score: {total_dice / num_images:.4f}")
print(f"Average mIoU      : {total_iou / num_images:.4f}")

print("=============================")
