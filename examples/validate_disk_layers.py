"""Explicit small ODE audit of saved multilayer quadrature intersections."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from numerical_observatory.disk_layers import scene_from_metadata
from numerical_observatory.layer_tracing import layer_task, LayerOrbit, LayerScene, collect_solution
from numerical_observatory.orbital_validation import pixel_momentum
from numerical_observatory.geodesic import trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("examples/disk_layer_analysis"))
    args = parser.parse_args()
    metadata = json.loads((args.input/"trace_metadata.json").read_text(encoding="utf-8"))
    metric, camera, original, scene = scene_from_metadata(metadata)
    points = [(310, 192), (266, 240), (285, 272), (260, 250), (379, 250), (250, 260),
              (389, 260), (275, 245), (280, 270), (319, 310), (320, 250), (0, 0)]
    if (camera.width, camera.height) != (640, 480) or metadata.get("samples", 1) != 1:
        raise ValueError("This explicit reference-point audit uses the native 640x480 example")
    cases = []
    started = time.perf_counter()
    with np.load(args.input/"disk_layers.npz") as data:
        for x, y in points:
            index = y*camera.width+x
            a = data["crossings"][data["ray_offsets"][index]:data["ray_offsets"][index+1]]
            runs = []
            for rtol in (original.rtol, min(original.rtol, 1e-11)):
                settings = replace(original, rtol=rtol, atol=min(original.atol, rtol/100))
                ode, _ = layer_task((metric, camera, settings, scene, metadata["collection"]["max_crossing"], "ode"), [index])
                b = ode[0][1]
                np.testing.assert_array_equal(a["plane_ordinal"], b["plane_ordinal"])
                np.testing.assert_array_equal(a["region"], b["region"])
                np.testing.assert_array_equal(a["emission_valid"], b["emission_valid"])
                valid = a["emission_valid"]
                run = dict(rtol=rtol, crossing_count=len(a), regions=a["region"].tolist(),
                           plane_ordinals=a["plane_ordinal"].tolist(), emitting_ordinals=a["emitting_ordinal"].tolist())
                if len(a):
                    run.update(max_spacetime_event_difference_M=float(np.max(np.linalg.norm(a["event"]-b["event"], axis=1)))/metric.mass,
                               max_affine_difference_M=float(np.max(abs(a["affine"]-b["affine"])))/metric.mass,
                               max_covector_difference=float(np.max(np.linalg.norm(a["momentum"]-b["momentum"], axis=1))),
                               max_orbital_angle_difference_radians=float(np.max(abs(a["phi_radians"]-b["phi_radians"]))))
                if valid.any():
                    g_error = float(np.max(abs(a["g"][valid]-b["g"][valid])))
                    intensity_error = float(np.max(abs(a["bolometric_intensity"][valid]/b["bolometric_intensity"][valid]-1)))
                    # Independent explicit circular-orbit expression for g.
                    g_reference = []
                    for row in a[valid]:
                        event, p = row["event"], row["momentum"]
                        radius = np.linalg.norm(event[1:])
                        omega = scene.motion.rotation*np.sqrt(metric.mass/radius**3)
                        ut = 1/np.sqrt(1-3*metric.mass/radius)
                        l_disk = np.dot(scene.surface.normal, np.cross(event[1:], p[1:]))
                        nu_obs = p[0]/np.sqrt(1-2*metric.mass/np.linalg.norm(camera.position))
                        g_reference.append(nu_obs/(ut*(p[0]+omega*l_disk)))
                    run.update(max_g_difference=g_error, max_relative_bolometric_difference=intensity_error,
                               max_explicit_formula_g_difference=float(np.max(abs(a["g"][valid]-g_reference))))
                runs.append(run)
            cases.append(dict(x=x, y=y, runs=runs))
    maxima = {}
    for case in cases:
        for key, value in case["runs"][-1].items():
            if key.startswith("max_"):
                maxima[key] = max(maxima.get(key, 0), value)
    assert maxima["max_spacetime_event_difference_M"] < 1e-6
    assert maxima["max_covector_difference"] < 1e-7
    assert maxima["max_g_difference"] < 1e-8
    assert maxima["max_relative_bolometric_difference"] < 1e-6
    assert maxima["max_explicit_formula_g_difference"] < 1e-12
    probes = []
    tight = replace(original, rtol=min(original.rtol, 1e-11), atol=min(original.atol, 1e-13))
    scene_state = LayerScene(scene.surface, scene.motion, scene.emission, scene.display, camera.tetrad(metric)[0])
    for x, y in ((319., 191.), (319., 192.), (319.5, 191.25), (319.5, 191.5), (319.5, 191.75)):
        p = pixel_momentum(camera, metric, x, y)
        orbit = LayerOrbit(metric, camera.event, p, original)
        hits, _, _ = orbit.crossings(scene.surface, orbit.transverse, 4)
        _, sol = trace(metric, camera.event, p, tight, return_solution=True)
        rows, *_ = collect_solution(metric, camera.event, p, tight, sol, scene_state, 4)
        hit = next((hit for hit in hits if hit.crossing_index == 2), None)
        assert hit is not None and len(rows) >= 2
        radius = np.linalg.norm(hit.event[1:])/metric.mass
        assert abs(radius-rows[1]["radius_M"]) < 1e-5
        probes.append(dict(x=x, y=y, plane_ordinal=2, radius_M=radius, emission_valid=bool(scene.surface.contains(hit.event)),
                           ode_radius_difference_M=float(abs(radius-rows[1]["radius_M"]))))
    report = dict(cases=cases, tight_run_maxima=maxima, subpixel_crown_probes=probes, elapsed_seconds=time.perf_counter()-started,
                  ode_integrations=2*len(cases)+len(probes), image_rerendered=False,
                  note="Explicit small audit of all recorded crossings, not a second image trace; first and later emitting events checked")
    (args.input/"validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(maxima, indent=2))


if __name__ == "__main__":
    main()
