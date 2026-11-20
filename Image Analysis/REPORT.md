# Spatial Image Filtering

## Problem statement

Spatial filtering modifies image pixels from their local neighborhoods. It is
used to reduce noise, sharpen soft details, and expose boundaries before later
analysis. A filter that removes noise too aggressively can also blur handwriting,
road markings, leaf margins, or product details. This mini-project implements
eight representative tasks using OpenCV, NumPy, Matplotlib, and public
scikit-image sample images. Each case saves its input, processed images, and a
labeled comparison figure. The Case 8 experiment additionally measures mean
squared error (MSE) and peak signal-to-noise ratio (PSNR).

The medical-style filtering exercise is an image-processing demonstration only.
It is not a clinical task and must not be used to make diagnostic decisions.
Replaceable test images should be public or otherwise used with permission.

## Objectives

1. Implement spatial sharpening, smoothing, morphological min/max, and edge
   operators using standard Python image-processing packages.
2. Compare methods using identical images and reproducible synthetic noise.
3. Examine the influence of kernel size and the trade-off between denoising and
   detail preservation.
4. Save reproducible images, comparison figures, and quantitative measurements.

## Methodology and algorithms

Inputs default to public teaching images exposed by `skimage.data`; optional
case-specific files in `inputs/` replace those defaults. Grayscale conversion
is used for document, microscopy, road-edge, foliage-edge, old-photo, and
product-filtering cases. Images are stored as 8-bit values from 0 to 255.
Synthetic noise uses a fixed random seed so that reruns are comparable.

The results below refer to the batch script, `spatial_filtering_project.py`.
The `inputs/` folder contains replacement images for Cases 1, 3, and 5:
`case1_document.png`, `case3_road.jpg`, and `case5_leaf.jpg`. The other cases
use their built-in scikit-image samples: microscopy (`cell`), a launch scene
(`rocket`), a grayscale photograph (`camera`), coins (`coins`), and an
astronaut portrait (`astronaut`). The interactive frontend can also use its
single optional upload for whichever case is selected.

### Sharpening and denoising (Cases 1, 2, 4, and 6)

Laplacian sharpening combines the original with its second spatial derivative:
`I_sharp = I - alpha * Laplacian(I)`. Unsharp masking subtracts a blurred copy
from the original and adds a scaled detail layer:
`I_sharp = I + amount * (I - GaussianBlur(I))`. Both emphasize high-frequency
detail, and both can exaggerate noise or create halos if the strength is too
large. Unsharp masking offers a direct, adjustable detail amount.

The mean filter replaces each pixel by the neighborhood average. A Gaussian
filter uses distance-weighted averaging and is effective against mild,
approximately Gaussian noise. A median filter selects the neighborhood median,
which is robust to isolated black/white impulses. Bilateral filtering combines
spatial closeness with intensity similarity, reducing smoothing across strong
intensity boundaries. It is more computationally expensive than a Gaussian
blur.

Case 2 injects salt-and-pepper impulses in a microscopy-style grayscale image,
then compares mean, Gaussian, and median filters. Case 4 adds Gaussian noise to
an outdoor sample and compares Gaussian with bilateral smoothing. Case 6 adds
grain, denoises with median and Gaussian filtering, and applies a restrained
unsharp mask after median filtering. This order reduces the risk of sharpening
noise; the sharpening strength should still be tuned conservatively.

### Edge detection (Cases 3 and 5)

The Sobel operator estimates horizontal and vertical first derivatives with
3x3 kernels. The Prewitt operator uses uniform 3x3 derivative kernels. Their
magnitude is computed as `sqrt(Gx^2 + Gy^2)`. A Gaussian pre-smoothing step
reduces sensitivity to pixel-level noise. Case 3 applies the two gradient
operators to a road scene. Case 5 compares Sobel magnitude with the absolute
Laplacian response for leaf/foliage boundaries.

Sobel generally gives clear, directional gradients and modest noise smoothing
through its kernel. Prewitt is simpler but can be less robust to noise. The
Laplacian responds to rapid intensity changes in every direction, but is
especially noise-sensitive and may create double edges; it has no edge
orientation information. Dense foliage texture can produce many responses, so
an isolated leaf image and thresholding/segmentation may be necessary for a
clean outline.

### Minimum and maximum filters (Case 7)

The local minimum is grayscale erosion; the local maximum is grayscale
dilation. Erosion suppresses small bright features and expands dark regions.
Dilation suppresses small dark features and expands bright regions. The
implementation compares 3x3, 5x5, and 7x7 square neighborhoods. Larger
neighborhoods have stronger effects and can remove small objects, thicken or
shrink boundaries, and erase fine detail. They should be selected with respect
to the feature size of interest.

### Quantitative comparison (Case 8)

The clean color sample is corrupted separately with additive Gaussian noise and
salt-and-pepper noise. Mean, Gaussian, median, and bilateral filters are
applied to each noisy image. For each result:

`MSE = mean((clean - filtered)^2)`

`PSNR = 10 * log10(255^2 / MSE)` dB.

Lower MSE and higher PSNR indicate closer pixel-wise agreement with the clean
reference, but these metrics do not always match perceived sharpness. The
script writes a row for each noise/filter pair to `outputs/case8_metrics.csv`
and plots PSNR in `outputs/figures/case8_psnr_comparison.png`. Inspect that
generated table and chart alongside `case8_filter_results.png`; the exact
ranking is image- and noise-dependent.

## Case-by-case answers and results

### Case 1 — Document sharpening

**Question:** Which sharpening method makes the blurred document more legible?

**Answer:** Unsharp masking is the preferred technique for readability in this
comparison. It increases local contrast around letter strokes while retaining
the broader page tones, giving a more balanced result. Laplacian sharpening
also improves edge visibility, but can emphasize fine texture and make strokes
look harsher. The visual difference is modest, and neither method restores
detail lost to blur.

**Method:** The grayscale image is sharpened with a 3x3 Laplacian at strength
0.7 and with an unsharp mask using a Gaussian blur (sigma 1.2) and amount 1.5.

**Figure:** `outputs/figures/case1_sharpening.png`

### Case 2 — Salt-and-pepper noise removal

**Question:** Which filter best removes isolated black and white impulse noise?

**Answer:** The 5x5 median filter gives the cleanest result in the comparison.
It removes most isolated impulses while preserving the sample boundary better
than mean or Gaussian averaging. Mean and Gaussian filtering reduce the noise
but spread impulse values into neighboring pixels, making the result look more
blurred.

**Method:** Six percent salt-and-pepper noise is added to a grayscale
microscopy sample. Mean, Gaussian (5x5, sigma 1.2), and median (5x5) filters
are compared.

**Figure:** `outputs/figures/case2_denoising.png`

### Case 3 — Edge detection

**Question:** Which operator produces clearer road boundaries in your selected
image?

**Answer:** On the selected road photograph, Sobel produces slightly stronger
and more continuous outlines around the painted lane markings and road edge,
so Sobel is clearer in this example. Prewitt detects the same markings, but
its lines appear somewhat weaker. Both operators also detect unrelated
details, including the bicycle symbol, fences, and buildings; neither
isolates road boundaries. This is a visual result for one image, not evidence
that Sobel is always superior. Gaussian pre-smoothing reduces small intensity
fluctuations before either operator is applied.

**Method:** Grayscale conversion, 5x5 Gaussian smoothing (sigma 1.0), then
3x3 Sobel and Prewitt gradient magnitudes. The selected input is a public-domain
cycle-lane street photograph; its source is credited in `README.md`.

**Figure:** `outputs/figures/case3_road_edges.png`

### Case 4 — Denoising a noisy outdoor scene

**Question:** Which filter reduces Gaussian noise while retaining boundaries?

**Answer:** Both filters reduce the added noise. In the comparison, the
bilateral result retains the launch vehicle and support-structure boundaries
more clearly than the Gaussian result, while leaving some residual texture.
The Gaussian filter smooths more uniformly and visibly softens edges. The
bilateral filter is preferable here when boundary preservation is important,
although it requires more computation.

**Method:** Gaussian noise with standard deviation 20 is added to a color
launch-scene sample. It is filtered with a 5x5 Gaussian blur (sigma 1.2) and a
bilateral filter (diameter 9; color and space sigmas 60).

**Figure:** `outputs/figures/case4_outdoor_denoising.png`

### Case 5 — Leaf outline edge detection

**Question:** Which operator best highlights the leaf outline, and what are its
limitations?

**Answer:** On the selected isolated-leaf photograph, Sobel highlights the
outer leaf outline more clearly and continuously than the Laplacian. It also
detects the internal veins. The Laplacian detects the outline too, but its
response is weaker in this comparison and it can produce double edges or
respond strongly to fine texture. Neither operator separates the leaf from
all other structures automatically; results depend on contrast, lighting, and
the selected thresholds. This is a visual comparison on one image, not a
claim that Sobel is always superior. The image is “Single leaf” by Titus
Tscharntke, in the public domain (source credited in `README.md`).

**Method:** Grayscale conversion and 5x5 Gaussian pre-smoothing (sigma 1.0),
followed by Sobel magnitude and a 3x3 Laplacian response.

**Figure:** `outputs/figures/case5_leaf_edges.png`

### Case 6 — Denoising followed by sharpening

**Question:** Does denoising before sharpening improve the result?

**Answer:** Yes. The median-filtered image followed by a moderate unsharp mask
reduces synthetic grain while restoring some edge definition. The Gaussian
result is smoother but loses more fine detail. The enhanced result is a
reasonable balance for this experiment, though some grain remains and
sharpening cannot recover information absent from the source.

**Method:** Gaussian noise with standard deviation 12 is added to a grayscale
photograph. A 3x3 median filter and a 5x5 Gaussian filter (sigma 1.0) are
compared; the median result is sharpened with an unsharp-mask amount of 1.1.
This is a synthetic-grain demonstration, not restoration of a scanned old
photograph.

**Figure:** `outputs/figures/case6_old_photo.png`

### Case 7 — Minimum and maximum filters

**Question:** How does neighborhood size affect minimum and maximum filtering?

**Answer:** The minimum filter (erosion) reduces bright regions, while the
maximum filter (dilation) expands them. Increasing the square neighborhood
from 3x3 to 5x5 and 7x7 makes the changes more pronounced and removes more
small detail. A 3x3 neighborhood is the least destructive of the tested
choices; larger sizes are useful only when their stronger morphological effect
matches the feature size of interest.

**Method:** Grayscale coin sample processed with minimum/maximum filters at
3x3, 5x5, and 7x7.

**Figure:** `outputs/figures/case7_min_max_filters.png`

### Case 8 — Quantitative denoising comparison

**Question:** Which filter performs best for Gaussian noise and for
salt-and-pepper noise?

**Answer:** On the clean astronaut sample and noise levels used here,
bilateral filtering performs best for Gaussian noise, with the lowest MSE and
highest PSNR. Median filtering performs best for salt-and-pepper noise. The
bilateral filter performs worst on the impulse-noise test in this run, showing
that filter choice depends on the noise type.

| Noise type | Best filter | MSE | PSNR (dB) |
|---|---|---:|---:|
| Gaussian | Bilateral | 57.3096 | 30.5485 |
| Salt-and-pepper | Median | 103.9134 | 27.9641 |

The full table contains results for all four filters and both noise types in
`outputs/case8_metrics.csv`; the visual and PSNR comparisons are in
`outputs/figures/case8_filter_results.png` and
`outputs/figures/case8_psnr_comparison.png`. Lower MSE and higher PSNR indicate
closer pixel-wise agreement with the clean reference, but they do not capture
every perceptual difference.

## Conclusion

No single spatial filter is best for every task. Median filtering is a strong
choice for salt-and-pepper noise; Gaussian smoothing is a simple choice for
random Gaussian noise; bilateral filtering can retain strong boundaries while
suppressing noise; and unsharp masking can improve mildly blurred detail when
applied after denoising. Sobel and Prewitt are useful first-derivative edge
operators, while Laplacian responses are direction-independent but more
noise-sensitive. Filter size and strength must be matched to the image and the
size of relevant features. Run `spatial_filtering_project.py` to generate the
actual images, chart, and MSE/PSNR table for the selected samples.

## Reproducibility and deliverables

Run the commands in [README.md](README.md). The script creates original-image
copies, processed outputs, and comparison figures under `outputs/`, including
the CSV and chart for Case 8. Replacing a default image only requires placing a
permitted file with the matching basename in `inputs/`.
