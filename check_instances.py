from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

mask = Image.open(
    r"E:\Goblet Cell\binary_test_masks\fbs02_0.png"
)

mask = np.array(mask)

labeled_mask, num_cells = ndimage.label(mask > 0)

print("Connected Components:", num_cells)

plt.figure(figsize=(6,6))
plt.imshow(labeled_mask)
plt.title(f"Cells = {num_cells}")
plt.show()