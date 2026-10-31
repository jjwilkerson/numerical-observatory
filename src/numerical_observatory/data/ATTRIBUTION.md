# Astronomical background

## CIE color-matching functions

`CIE_xyz_1931_2deg.csv` is the unmodified CIE 1931 2° standard observer table,
360–830 nm in 1 nm steps (CIE 018:2019, Table 6). Its original metadata is
bundled as `CIE_xyz_1931_2deg.csv_metadata.json`.

Citation: **CIE 2019, Colour-matching functions of CIE 1931 standard colorimetric
observer, International Commission on Illumination (CIE), Vienna, AT**,
DOI: https://doi.org/10.25039/CIE.DS.xvudnb9b.

- Source: https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
- Data: https://files.cie.co.at/Publications-datasets/CIE_xyz_1931_2deg.csv
- License: **Creative Commons Attribution-ShareAlike 4.0 International**,
  https://creativecommons.org/licenses/by-sa/4.0/ (see bundled metadata).
- MD5: `17cca777db64b17170f06f67ce9d3ab7`.
- SHA256: `fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1`.

The numerical temperature/chromaticity lookup is generated in memory from this
table using Planck spectra. The source table has no edits or resampling. The
derived lookup is attributed to CIE and shared under the same CC BY-SA 4.0
license. No CIE endorsement of the renderer or its images is implied.

## NASA star maps

`starmap_g4k.jpg` and `starmap_g8k.jpg` are the unmodified 4096×2048 and
8192×4096 galactic-coordinate star maps from
NASA's **Deep Star Maps** (2012), created by Ernie Wright and Tom Bridgman.

Credit: **NASA/Goddard Space Flight Center Scientific Visualization Studio**.

- Source and catalog information: https://svs.gsfc.nasa.gov/3895/
- Original file: https://svs.gsfc.nasa.gov/vis/a000000/a003800/a003895/starmap_g4k.jpg
- Higher-resolution file: https://svs.gsfc.nasa.gov/vis/a000000/a003800/a003895/starmap_g8k.jpg
- NASA media guidelines: https://www.nasa.gov/nasa-brand-center/images-and-media/

The map visualizes positions, brightnesses, and colors of more than 100 million
stars from the Bright Star, Tycho-2, and UCAC3 catalogs. It is a catalog-based
visualization, not a single telescope photograph. No constellation overlays
are included. The 2012 JPEG keeps runtime dependencies limited to Pillow and
NumPy; newer 2020 full-resolution maps require OpenEXR decoding.

The observatory rotates and samples this map along geodesic source directions
and applies a configurable display exposure. The results are derived
visualizations, not NASA images of a black hole or calibrated photometry.
NASA permits educational/informational computer graphical simulations under its
media guidelines. No NASA endorsement is implied.
