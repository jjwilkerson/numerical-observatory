"""Audit the specific 640x480 edge-on example from saved values only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from PIL import Image, ImageDraw

from numerical_observatory.disk_diagnostics import extract


def circular_span(angles):
    values = np.sort(np.asarray(angles) % 360)
    if not len(values):
        return None
    gaps = np.diff(np.r_[values, values[0]+360])
    largest = int(np.argmax(gaps))
    start = float(values[(largest+1) % len(values)])
    span = float(360-gaps[largest])
    return dict(start_degrees=start, span_degrees=span, end_degrees=(start+span) % 360,
                occupied_10_degree_bins=int(np.unique(np.floor(values/10)).size),
                note="Shortest circular interval covering samples; not proof of continuous coverage between samples")


def statistics(arrays, mask):
    valid = mask & arrays["radius_valid"] & arrays["azimuth_valid"] & arrays["crossing_valid"]
    groups = {}
    for k in np.unique(arrays["crossing_number"][valid]):
        selected = valid & (arrays["crossing_number"] == k)
        groups[str(int(k))] = dict(count=int(selected.sum()),
            radius_m_range=[float(arrays["radius_m"][selected].min()), float(arrays["radius_m"][selected].max())],
            azimuth=circular_span(arrays["azimuth_degrees"][selected]))
    return dict(emitting_samples=int(valid.sum()), by_crossing=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=Path("examples/disk_diagnostics"))
    args = parser.parse_args()
    output = args.diagnostics
    metadata = json.loads((output/"metadata.json").read_text(encoding="utf-8"))
    if metadata["source_image_size"] != [640, 480] or metadata["samples_per_axis"] != 1:
        raise ValueError("These explicit visual ROIs apply only to the saved 640x480, one-ray-per-pixel example")
    with np.load(output/"diagnostic_arrays.npz") as data:
        arrays = {key: data[key] for key in data.files}
    yy, xx = np.indices(arrays["status"].shape)
    distance = np.hypot(xx-319.5, yy-239.5)
    masks = dict(upper_halo=(distance >= 46) & (distance <= 50) & (yy < 232),
                 lower_halo=(distance >= 46) & (distance <= 50) & (yy >= 264),
                 left_branch=(xx >= 260) & (xx < 278) & (yy >= 220) & (yy < 244),
                 right_branch=(xx >= 362) & (xx < 380) & (yy >= 220) & (yy < 244),
                 upper_thin_crossing_2=(arrays["crossing_number"] == 2) & (yy < 244))
    definitions = dict(upper_halo="46<=image distance from (319.5,239.5)<=50 pixels, y<232",
                       lower_halo="46<=image distance from (319.5,239.5)<=50 pixels, y>=264",
                       left_branch="260<=x<278, 220<=y<244",
                       right_branch="362<=x<380, 220<=y<244",
                       upper_thin_crossing_2="all recorded crossing-2 emitting rays with y<244; categorically selected")
    report = dict(roi_definitions=definitions, regions={name: statistics(arrays, mask) for name, mask in masks.items()},
                  warning="Image-space ROIs select visible features only. Physical values always come from recorded hits. Branch boxes also overlap foreground crossing-1 disk.")
    report["upper_thin_crossing_2_8_connected_components"] = int(label(masks["upper_thin_crossing_2"], structure=np.ones((3, 3)))[1])
    report["branch_row_samples"] = {}
    for side in ("left", "right"):
        mask = masks[side+"_branch"] & (arrays["crossing_number"] == 2)
        rows = []
        for y in np.unique(np.where(mask)[0]):
            x = np.flatnonzero(mask[y])
            phi = arrays["azimuth_degrees"][y, x]
            mean = float(np.degrees(np.angle(np.exp(1j*np.deg2rad(phi)).mean())) % 360)
            rows.append(dict(y=int(y), x=x.tolist(), radius_m=arrays["radius_m"][y, x].tolist(),
                             azimuth_degrees=phi.tolist(), circular_mean_degrees=mean))
        continuous = np.rad2deg(np.unwrap(np.deg2rad([row["circular_mean_degrees"] for row in rows])))
        report["branch_row_samples"][side] = dict(rows=rows,
            max_neighbor_row_azimuth_mean_change_degrees=float(np.max(np.abs(np.diff(continuous)))))
    source = Path(metadata["input_render"])
    with np.load(metadata["source_rays"]) as raw:
        report["representative_saved_hits"] = []
        normal = np.array(metadata["extraction"]["basis"]["normal"])
        for x, y in ((310, 192), (313, 192), (286, 205), (287, 205), (266, 240), (269, 240),
                     (373, 240), (370, 240), (284, 272), (285, 272), (311, 287)):
            position = raw["hit_position"][y, x]
            report["representative_saved_hits"].append(dict(x=x, y=y, status=int(arrays["status"][y, x]),
                radius_m=float(arrays["radius_m"][y, x]), crossing=int(arrays["crossing_number"][y, x]),
                azimuth_degrees=float(arrays["azimuth_degrees"][y, x]), position=position.tolist(),
                plane_residual=float(abs(position @ normal))))
    serial = Path("examples/disks/disk-edge-on-hires")
    if (serial/"disk_rays.npz").exists():
        old_metadata = json.loads((serial/"diagnostics.json").read_text(encoding="utf-8"))
        with np.load(serial/"disk_rays.npz") as old:
            original_arrays, _ = extract(old, old_metadata)
        for name in original_arrays:
            np.testing.assert_array_equal(original_arrays[name], arrays[name])
        report["serial_parallel_640_comparison"] = "All extracted fields exactly equal between earlier serial thermal and parallel blackbody saved runs"
    np.savez_compressed(output/"feature_masks.npz", **masks)
    (output/"feature_statistics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with Image.open(source/"shadow.png") as image:
        marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    draw.ellipse((269, 189, 370, 290), outline=(255, 100, 70), width=1)
    draw.rectangle((260, 220, 277, 243), outline=(0, 255, 255), width=1)
    draw.rectangle((362, 220, 379, 243), outline=(0, 255, 255), width=1)
    draw.text((390, 195), "Thin-halo annulus: R=46..50 px", fill="white")
    draw.text((390, 213), "Cyan boxes: branch ROIs", fill="white")
    marked.save(output/"region_selection.png")
    print(json.dumps({name: value for name, value in report.items() if name in ("regions", "upper_thin_crossing_2_8_connected_components", "serial_parallel_640_comparison")}, indent=2))


if __name__ == "__main__":
    main()
