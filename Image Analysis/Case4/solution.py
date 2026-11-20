from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "Sample_Image.png"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

image_bgr = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
rng = np.random.default_rng(2026)
noisy = np.clip(
    image.astype(np.float32) + rng.normal(0.0, 20.0, image.shape), 0, 255
).astype(np.uint8)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 1.2)
bilateral = cv2.bilateralFilter(noisy, 9, 60, 60)

for name, result in (
    ("original.png", image), ("gaussian_noisy.png", noisy),
    ("gaussian_filtered.png", gaussian),
    ("bilateral_filtered.png", bilateral),
    ("preferred_result.png", bilateral),
):
    if not cv2.imwrite(
        str(OUTPUT / name), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"Could not save image: {OUTPUT / name}")

panels = [
    ("Original", image), ("Gaussian-noisy input", noisy),
    ("Gaussian filter 5x5", gaussian), ("Bilateral filter", bilateral),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (title, result) in zip(axes.flat, panels):
    ax.imshow(result)
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("Case 4: Filtering a noisy outdoor image")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 4 results saved under: {OUTPUT}")
