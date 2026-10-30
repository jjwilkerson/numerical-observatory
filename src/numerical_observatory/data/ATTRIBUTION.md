# Astronomical background

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
