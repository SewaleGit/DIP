from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT.parent / "Blurred_Sample_Image.jpeg"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

image_bgr = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
laplacian_sharpened = np.clip(
    gray.astype(np.float32) - 0.7 * laplacian, 0, 255
).astype(np.uint8)
softened = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
unsharp_masked = cv2.addWeighted(gray, 2.5, softened, -1.5, 0)

for name, result in (
    ("original_grayscale.png", gray),
    ("laplacian_sharpened.png", laplacian_sharpened),
    ("unsharp_masked.png", unsharp_masked),
):
    if not cv2.imwrite(str(OUTPUT / name), result):
        raise OSError(f"Could not save image: {OUTPUT / name}")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, title, result in zip(
    axes,
    ("Original grayscale", "Laplacian sharpened", "Unsharp mask"),
    (gray, laplacian_sharpened, unsharp_masked),
):
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 1 results saved under: {OUTPUT}")