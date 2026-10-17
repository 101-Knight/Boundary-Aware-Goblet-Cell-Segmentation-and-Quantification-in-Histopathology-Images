from PIL import Image
import numpy as np

mask = Image.open(
    r"E:\Goblet Cell\binary_test_masks\fbs02_0.png"
)

mask = np.array(mask)

print("Min:", mask.min())
print("Max:", mask.max())
print("Unique values:", np.unique(mask))