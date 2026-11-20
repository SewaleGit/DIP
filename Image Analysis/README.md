# Spatial Image Filtering Mini-Project

This project contains runnable Python solutions for all eight spatial-filtering
cases in the assignment. The script uses public sample images provided by
scikit-image when no replacement image is supplied. Synthetic noise is
reproducible (fixed random seed), and generated deliverables are saved beneath
`outputs/`.

## Setup and run

```powershell
py -m pip install opencv-python numpy matplotlib scikit-image
py spatial_filtering_project.py
```

If `py` is not available, use `python` in its place. The script generates:

- `outputs/originals/`: copies of the input images.
- `outputs/results/`: processed images from each case.
- `outputs/figures/`: labeled side-by-side comparisons and the Case 8 PSNR chart.
- `outputs/case8_metrics.csv`: MSE and PSNR for each filter/noise pair.

## Interactive frontend

Install Streamlit and start the browser interface:

```powershell
py -m pip install streamlit
py -m streamlit run app.py
```

The frontend lets you choose any of the eight cases and optionally upload an
image. Without an upload, it uses that case's matching public sample or an
image with the matching basename in `inputs/`. It opens in a local browser and
displays the processed image panels and Case 8 metrics interactively. In Case 1,
select **Save four-panel PNG** to save a labeled, single-row comparison image
to `Case1/outputs/Compared.png`.

The images from `skimage.data` are public teaching samples. Case 5 uses the
included isolated-leaf photograph by default; replace it with a permitted image
as described below if needed. The medical-style sample is microscopy imagery
for a filtering demonstration, not a clinical image or diagnostic output.

## Bring your own images

Create an `inputs` folder next to the script and add any of these files (PNG,
JPG/JPEG, TIFF, or BMP). Unspecified cases continue to use their public samples.

| Case | Expected basename | Suggested image |
|---|---|---|
| 1 | `case1_document` | Handwritten or printed document |
| 2 | `case2_medical` | Public, non-clinical microscopy image |
| 3 | `case3_road` | Road photograph |
| 4 | `case4_cctv` | Outdoor/CCTV-style scene |
| 5 | `case5_leaf` | Leaf photograph |
| 6 | `case6_old_photo` | Photograph with grain/artifacts |
| 7 | `case7_product` | Product image with small details |
| 8 | `case8_clean` | Clean image suitable for controlled noise tests |

Case 5 includes an isolated leaf photograph at `inputs/case5_leaf.jpg`. In this
example, Sobel shows the outer outline more clearly than the Laplacian, while
both also detect internal veins. The photograph, “Single leaf” by Titus
Tscharntke, is public domain on
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Single_leaf.jpg).

Case 3 includes a road photograph with visible painted lane markings at
`inputs/case3_road.jpg`. In this example, Sobel traces the markings slightly
more clearly than Prewitt, although both also detect surrounding scene details.
The photo is “Cycle lane in Excalibur Drive, Cardiff” by Seth Whales,
available in the public domain on
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Cycle_lane_in_Excalibur_Drive,_Cardiff.jpg).
If the file is removed, the script falls back to a motorcycle scene, which is
only suitable for comparing general scene edges.

Use images you have permission to use. Inputs are loaded as color images and
converted to grayscale for cases that require it.

## Method and interpretation

The implementation applies Laplacian and unsharp-mask sharpening (Case 1);
mean, Gaussian, and median smoothing (Case 2); Sobel and Prewitt edge detection
(Case 3); Gaussian and bilateral denoising (Case 4); Sobel and Laplacian edge
detection (Case 5); median/Gaussian denoising followed by unsharp masking
(Case 6); 3x3, 5x5, and 7x7 minimum/maximum filters (Case 7); and four
denoisers evaluated with MSE and PSNR for Gaussian and impulse noise (Case 8).

For salt-and-pepper noise, the median filter is expected to perform best because
it rejects isolated extreme pixels without averaging them into their
neighborhoods. The bilateral filter can preserve strong boundaries better than
a Gaussian blur because its weights depend on both spatial distance and
intensity similarity. For Case 8, use both the visual comparison and the
generated metrics: median filtering is typically strongest for impulse noise;
Gaussian filtering often works well for Gaussian noise, while bilateral
filtering may preserve edges at the cost of more computation. The actual
ranking depends on image content and noise strength.

See [REPORT.md](REPORT.md) for the short report, algorithm details, limitations,
and discussion of the results.
