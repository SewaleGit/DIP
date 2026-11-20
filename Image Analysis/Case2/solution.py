from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

image = data.cell()

rng = np.random.default_rng(2026)
noisy = image.copy()
mask = rng.random(image.shape)
noisy[mask < 0.03] = 0
noisy[(mask >= 0.03) & (mask < 0.06)] = 255

results = {
    "mean_5x5.png": cv2.blur(noisy, (5, 5)),
    "gaussian_5x5.png": cv2.GaussianBlur(noisy, (5, 5), 1.2),
    "median_5x5.png": cv2.medianBlur(noisy, 5),
}
for name, result in (
    ("original.png", image),
    ("salt_pepper_noisy.png", noisy),
    *results.items(),
):
    if not cv2.imwrite(str(OUTPUT / name), result):
        raise OSError(f"Could not save image: {OUTPUT / name}")

panels = [("Clean sample", image), ("Salt-and-pepper noise", noisy)]
panels.extend((name.removesuffix(".png").replace("_", " ").title(), result)
              for name, result in results.items())
fig, axes = plt.subplots(1, len(panels), figsize=(20, 5))
for ax, (title, result) in zip(axes, panels):
    ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("Case 2: Salt-and-pepper noise removal")
fig.tight_layout()
fig.savefig(OUTPUT / "comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Case 2 results saved under: {OUTPUT}")
