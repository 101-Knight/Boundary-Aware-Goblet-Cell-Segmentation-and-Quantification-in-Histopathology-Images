import cv2
import numpy as np
from scipy import ndimage

def count_goblet_cells(mask):

    mask = (mask > 0).astype(np.uint8)

    labeled_mask, num_cells = ndimage.label(mask)

    return num_cells, labeled_mask