from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

mask = Image.open(
    r"E:\Goblet Cell\binary_test_masks\fbs02_0.png"
)

mask = np.array(mask)

binary = mask > 0

distance = ndimage.distance_transform_edt(binary)

coords = peak_local_max(
    distance,
    footprint=np.ones((15, 15)),
    labels=binary
)

markers = np.zeros_like(distance, dtype=int)

for i, coord in enumerate(coords):
    markers[coord[0], coord[1]] = i + 1

labels = watershed(
    -distance,
    markers,
    mask=binary
)

print("Estimated Cells:", labels.max())

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("Binary Mask")
plt.imshow(binary)

plt.subplot(1,2,2)
plt.title(f"Watershed Cells = {labels.max()}")
plt.imshow(labels)

plt.show()