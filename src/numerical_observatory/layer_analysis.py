"""Raster and source-coordinate evidence for a revealed layer; no ray tracing."""
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label


def summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(count=int(finite.size), minimum=float(finite.min()), median=float(np.median(finite)),
                p95=float(np.percentile(finite, 95)), maximum=float(finite.max())) if finite.size else dict(count=0)


def angular_distance(a, b):
    return abs((a-b+180) % 360-180)


def adjacency_changes(grid, boundary_only=False):
    """Four unique directions cover undirected 8-neighbor edges exactly once."""
    radius, azimuth, hidden, valid = (grid[key] for key in ("radius_M", "azimuth_degrees", "concealed", "valid"))
    dr, da = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        h, w = valid.shape
        ya, yb = slice(0, h-dy), slice(dy, h)
        xa, xb = (slice(0, w-dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w+dx))
        mask = valid[ya, xa] & valid[yb, xb]
        if boundary_only:
            mask &= hidden[ya, xa] != hidden[yb, xb]
        dr.append(abs(radius[ya, xa][mask]-radius[yb, xb][mask]))
        da.append(angular_distance(azimuth[ya, xa][mask], azimuth[yb, xb][mask]))
    return dict(radius_neighbor_difference_M=summary(np.concatenate(dr)),
                circular_azimuth_neighbor_difference_degrees=summary(np.concatenate(da)),
                interpretation="Sampled gradients only; equal ordinal or raster adjacency does not prove a continuous physical mapping")


def shortest_path(mask, start, goal):
    """8-connected path through actual sampled emitting pixels only."""
    height, width = mask.shape
    queue = deque([start])
    previous = {start: None}
    while queue:
        y, x = queue.popleft()
        if (y, x) == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node[::-1])
                node = previous[node]
            return np.array(path[::-1], dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbor = y+dy, x+dx
                if (dy or dx) and 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and mask[neighbor] and neighbor not in previous:
                    previous[neighbor] = (y, x)
                    queue.append(neighbor)
    return np.empty((0, 2), np.int32)


def analyze(data, selected, metadata, output, target, features=None):
    output = Path(output)
    complete, count = label(selected["valid"], structure=np.ones((3, 3)))
    visible, visible_count = label(selected["normally_visible"], structure=np.ones((3, 3)))
    sizes = np.bincount(complete.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax()) if count else 0
    main = (complete == largest) & selected["valid"]
    report = dict(target_crossing=target, complete_samples=int(selected["valid"].sum()),
                  normally_visible_samples=int(selected["normally_visible"].sum()), concealed_samples=int(selected["concealed"].sum()),
                  visible_components_8=int(visible_count), complete_components_8=int(count),
                  largest_component=int(largest), largest_component_samples=int(sizes[largest]) if count else 0,
                  concealed_samples_in_largest=int((main & selected["concealed"]).sum()),
                  all_adjacent_samples=adjacency_changes(selected), visibility_boundary_edges=adjacency_changes(selected, True),
                  source_radius_M=summary(selected["radius_M"][selected["valid"]]),
                  source_azimuth_degrees=summary(selected["azimuth_degrees"][selected["valid"]]),
                  note="No pixels are dilated, interpolated, filled, or reclassified for this analysis. Connectivity is sampled-raster connectivity only.")
    arrays = dict(component_labels=complete, visible_component_labels=visible, main_component=main)
    groups = {}
    feature_path = None if features is None else Path(features)
    if target == 2 and selected["valid"].shape == (480, 640) and feature_path is not None and (feature_path/"feature_masks.npz").exists():
        with np.load(feature_path/"diagnostic_arrays.npz") as old:
            source_cross = old["crossing_number"]
            expected = (old["status"] == 3) & (source_cross == 2)
            if not np.array_equal(expected, selected["normally_visible"]):
                report["feature_warning"] = "Existing feature products are not the same opaque ray sample; region-specific analysis skipped."
            else:
                with np.load(feature_path/"feature_masks.npz") as masks:
                    groups = {key: masks[key] & (source_cross == 2) for key in ("upper_halo", "left_branch", "right_branch")}
                yy, xx = np.indices(expected.shape)
                groups["broad_lower_crescent"] = (source_cross == 2) & (yy >= 263) & (np.hypot(xx-319.5, yy-239.5) >= 50)
    report["regions"] = {}
    for name, mask in groups.items():
        chosen = mask & selected["valid"]
        report["regions"][name] = dict(samples=int(chosen.sum()), samples_in_main_component=int((chosen & main).sum()),
            component_labels=[int(v) for v in np.unique(complete[chosen])],
            radius_M=summary(selected["radius_M"][chosen]), azimuth_degrees=summary(selected["azimuth_degrees"][chosen]))
        arrays[name] = chosen
    report["bridge_paths"] = {}
    left_path = None
    if groups:
        for side in ("left", "right"):
            candidates = np.argwhere(groups[side+"_branch"] & main)
            if not len(candidates):
                continue
            # Start at the highest connected part of the existing branch mask.
            start = tuple(candidates[np.lexsort((candidates[:, 1], candidates[:, 0]))[0]])
            goal_requested = (280, 260 if side == "left" else 379)
            goals = np.argwhere(groups["broad_lower_crescent"] & main)
            goal = tuple(goals[np.argmin(np.sum((goals-goal_requested)**2, axis=1))])
            if side == "right" and left_path is not None and np.array_equal(selected["valid"], selected["valid"][:, ::-1]):
                path = left_path.copy()
                path[:, 0] = selected["valid"].shape[1]-1-path[:, 0]
            else:
                path = shortest_path(selected["valid"], start, goal)
                if side == "left":
                    left_path = path
            arrays[side+"_bridge_path_xy"] = path
            x, y = path.T
            radius, phi = selected["radius_M"][y, x], selected["azimuth_degrees"][y, x]
            hidden = selected["concealed"][y, x]
            hidden_edges = hidden[1:] & hidden[:-1]
            record = dict(pixel_count=len(path), hidden_pixels=int(selected["concealed"][y, x].sum()),
                radius_M=summary(radius), azimuth_degrees=phi.tolist(),
                azimuth_unwrapped_degrees=np.rad2deg(np.unwrap(np.deg2rad(phi))).tolist(),
                max_step_radius_M=float(np.max(abs(np.diff(radius)))),
                max_step_azimuth_degrees=float(np.max(angular_distance(phi[1:], phi[:-1]))),
                within_hidden_step_radius_M=summary(abs(np.diff(radius))[hidden_edges]),
                within_hidden_step_azimuth_degrees=summary(angular_distance(phi[1:], phi[:-1])[hidden_edges]),
                path_selection="Shortest sampled 8-neighbor path; right reflected from left when validity mask is exactly symmetric. No missing pixels filled.",
                samples=[dict(x=int(px), y=int(py), radius_M=float(selected["radius_M"][py, px]),
                              azimuth_degrees=float(selected["azimuth_degrees"][py, px]), concealed=bool(selected["concealed"][py, px]))
                         for px, py in path])
            # Assert the reported bridge never jumps a missing native sample.
            assert np.max(abs(np.diff(path, axis=0))) <= 1
            report["bridge_paths"][side] = record
    (output/"continuity.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(output/"continuity_arrays.npz", **arrays)
    return report
