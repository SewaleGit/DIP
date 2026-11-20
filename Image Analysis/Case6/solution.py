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
rng = np.random.default_rng(2026)
noisy = np.clip(
    image.astype(np.float32) + rng.normal(0.0, 12.0, image.shape), 0, 255
).astype(np.uint8)
median = cv2.medianBlur(noisy, 3)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.0)
softened = cv2.GaussianBlur(median, (0, 0), sigmaX=1.2)
enhanced = cv2.addWeighted(median, 2.1, softened, -1.1, 0)

for name, result in (
    ("original.png", image), ("synthetic_grain.png", noisy),
    ("median_denoised.png", median), ("gaussian_denoised.png", gaussian),
    ("preferred_enhanced.png", enhanced),
):
    if not cv2.imwrite(str(OUTPUT / name), result):
        raise OSError(f"Could not save image: {OUTPUT / name}")

panels = [
    ("Original", image), ("Synthetic grain", noisy),
    ("Median denoised", median), ("Gaussian denoised", gaussian),
    ("Median + unsharp mask", enhanced),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (title, result) in zip(axes.flat, panels):
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
for ax in axes.flat[len(panels):]:
    ax.axis("off")
fig.suptitle("Case 6: Denoising followed by controlled sharpening")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 6 results saved under: {OUTPUT}")
