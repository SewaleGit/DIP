from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "Sample_Image.png"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")
panels = [("Original grayscale", image)]
for size in (3, 5, 7):
    kernel = np.ones((size, size), dtype=np.uint8)
    minimum = cv2.erode(image, kernel)
    maximum = cv2.dilate(image, kernel)
    for kind, result in (("minimum", minimum), ("maximum", maximum)):
        name = f"{kind}_{size}x{size}.png"
        if not cv2.imwrite(str(OUTPUT / name), result):
            raise OSError(f"Could not save image: {OUTPUT / name}")
        panels.append((f"{kind.title()} {size}x{size}", result))
if not cv2.imwrite(str(OUTPUT / "original.png"), image):
    raise OSError(f"Could not save image: {OUTPUT / 'original.png'}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (title, result) in zip(axes.flat, panels):
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("Case 7: Minimum and maximum filters")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 7 results saved under: {OUTPUT}")
