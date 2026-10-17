from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

mask = Image.open(
    r"E:\Goblet Cell\GCSdataV1.1\original_data\train_test_split\patched_img_level\train\masks\fbs01_0.png"
)

mask = np.array(mask)

print("Shape:", mask.shape)
print("Min:", mask.min())
print("Max:", mask.max())

plt.imshow(mask, cmap="gray")
plt.colorbar()
plt.show()