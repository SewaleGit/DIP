"""Discrete Cosine Transform using the orthonormal DCT-II.
Read an image, apply the transform, and display the result.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from skimage import data
from skimage.transform import resize
from scipy.fft import dctn

# 1. Read image.jpg when it exists; otherwise use a built-in sample.
image_path = Path(__file__).with_name("image.jpg")
if image_path.is_file():
    image = imread(image_path, as_gray=True)
else:
    image = data.camera()

# 2. Resize to 128 by 128 pixels for this lab.
image = resize(image, (128, 128), anti_aliasing=True)

# 3. Apply the two-dimensional DCT.
transformed = dctn(image, type=2, norm="ortho")

# 4. Display the original image and the transformed image.
# Log magnitude makes small coefficients visible; it is only for display.
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(np.log1p(np.abs(transformed)), cmap="gray")
plt.title("DCT coefficients")
plt.axis("off")
plt.tight_layout()
output_path = Path(__file__).with_name("outputs") / "dct_transform.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved DCT figure to {output_path}")
