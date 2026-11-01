"""CPU rendering with optional spawn workers, portable output and diagnostics."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import scipy

from . import __version__
from .background import Starfield
from .symmetry import CenteredRayCache
from .metric import Schwarzschild
from .schwarzschild_orbit import compatible


def render(metric, camera, integration, output, progress=None, background=None,
           use_symmetry=True, domain=None, method="ode", samples=1, disk=None, workers=1):
    from .parallel import resolve_workers
    from .output import staged_render
    from .kerr_carter import validate_backend
    validate_backend(metric,integration,domain,None if disk is None else disk.surface)
    from .numerical_metric import NumericalMetric
    if type(metric) is NumericalMetric:
        if method!='ode' or disk is not None or domain is not None:
            raise ValueError('numerical metrics require guarded generic Hamiltonian, numerical domain, and no disk')
        from .numerical_trace import NumericalDomain
        NumericalDomain(metric).validate_camera(camera.event)
    from .binary_ks_approx import BinaryKSApprox
    from .binary_ks_orbit_approx import BinaryKSOrbitApprox
    if type(metric) in (BinaryKSApprox,BinaryKSOrbitApprox):
        if method!='ode' or disk is not None or domain is not None:
            raise ValueError('binary approximations support only generic Hamiltonian ODE, their built-in domain, and no disk')
        if type(metric) is BinaryKSOrbitApprox:
            from .orbit_binary_support import default_domain
        else:
            from .binary_support import default_domain
        default_domain(metric,camera.event,integration)
    selected = resolve_workers(workers)
    return staged_render(output, lambda stage: _render(metric, camera, integration, stage, progress,
                         background, use_symmetry, domain, method, samples, disk, selected))


def _render(metric, camera, integration, output, progress=None, background=None,
            use_symmetry=True, domain=None, method="ode", samples=1, disk=None, workers=1):
    if method not in ("ode", "quadrature"):
        raise ValueError("method must be ode or quadrature")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if disk is not None:
        from .kerr import Kerr
        if type(metric) is Kerr:
            from .surfaces import KerrEquatorialAnnulus
            from .emitter_motion import KerrCircular
            if (not isinstance(disk.surface, KerrEquatorialAnnulus) or not isinstance(disk.motion, KerrCircular)
                    or disk.surface.metric != metric or disk.motion.metric != metric):
                raise ValueError("Kerr rendering requires a matching equatorial Kerr disk, not Schwarzschild disk physics")
        from .surface_render import render_surface
        return render_surface(metric, camera, integration, output, disk, background, samples,
                              domain, progress, requested_method=method, use_symmetry=use_symmetry, workers=workers)
    if method == "quadrature" and compatible(metric, camera) and domain is None:
        from .diagnostic_render import render_quadrature
        return render_quadrature(metric, camera, integration, output,
                                 Starfield() if background is None else background, samples, workers=workers,
                                 progress=progress)
    if samples > 1:
        from .diagnostic_render import shade, save_image
        background = Starfield() if background is None else background
        sampled_camera = replace(camera, width=camera.width*samples, height=camera.height*samples)
        report = _render(metric, sampled_camera, integration, output, progress, background,
                         use_symmetry=use_symmetry, domain=domain, method="ode", workers=workers)
        output = Path(output)
        with np.load(output/"rays.npz") as rays:
            directions, status = rays["directions"], rays["status"]
            extra = {key: rays[key] for key in rays.files if key not in ("directions", "status", "samples")}
        save_image(output/"shadow.png", shade(directions, status, background, samples))
        np.savez_compressed(output/"rays.npz", directions=directions, status=status, samples=samples, **extra)
        report.update(camera=asdict(camera), samples_per_axis=samples, requested_method=method,
                      count_units="subpixel rays")
        (output/"diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return report
    # Validate the camera before allocating or tracing an image.
    camera.tetrad(metric)
    image = np.zeros((camera.height, camera.width, 3))
    background = Starfield() if background is None else background
    tracer = CenteredRayCache(metric, camera, integration, enabled=use_symmetry, domain=domain)
    if workers > 1:
        from .parallel import ParallelRays
        tracer = ParallelRays(metric, camera, integration, workers, use_symmetry, domain)
    directions = np.full((camera.height, camera.width, 3), np.nan)
    statuses = np.zeros((camera.height, camera.width), dtype=np.uint8)
    from .kerr import Kerr
    is_kerr = type(metric) is Kerr
    from .binary_ks_approx import BinaryKSApprox
    from .binary_ks_orbit_approx import BinaryKSOrbitApprox
    is_orbit_binary = type(metric) is BinaryKSOrbitApprox
    is_binary = type(metric) in (BinaryKSApprox,BinaryKSOrbitApprox)
    binary_arrays = ({'capture_index':np.zeros(statuses.shape,dtype=np.uint8),
                      'signature_margin':np.full(statuses.shape,np.nan),
                      'null_error':np.full(statuses.shape,np.nan),
                      'energy_drift':np.full(statuses.shape,np.nan),
                      'constraint_diagnostics_complete':np.zeros(statuses.shape,dtype=bool)} if is_binary else {})
    binary_reasons=np.empty(statuses.shape,dtype=object) if is_binary else None
    from .numerical_metric import NumericalMetric
    is_numerical=type(metric) is NumericalMetric
    numerical_arrays=({'null_error':np.full(statuses.shape,np.nan),'finest_selected_level':np.full(statuses.shape,-1,dtype=np.int32),
                       'selected_level_count':np.zeros(statuses.shape,dtype=np.uint16),
                       'capture_index':np.zeros(statuses.shape,dtype=np.uint16),'time_slabs':np.zeros(statuses.shape,dtype=np.uint32),
                       'temporal_stage_rejections':np.zeros(statuses.shape,dtype=np.uint32),
                       'lookup_stage_rejections':np.zeros(statuses.shape,dtype=np.int32),
                       'constraint_diagnostics_complete':np.zeros(statuses.shape,dtype=bool)} if is_numerical else {})
    numerical_reasons=np.empty(statuses.shape,dtype=object) if is_numerical else None
    level_counts=Counter();numerical_failures=Counter();numerical_metric_calls=numerical_stencils=0
    if is_orbit_binary:
        binary_arrays.update(energy_change=np.full(statuses.shape,np.nan),helical_error=np.full(statuses.shape,np.nan))
    from .backend_records import BackendRecords
    backend_records=BackendRecords((camera.height,camera.width),integration) if is_kerr else None
    conservation = ({key: np.full((camera.height, camera.width), np.nan)
                     for key in ("null_error", "energy_drift", "axial_momentum_drift", "carter_drift")}
                    if is_kerr else {})
    constants = np.full((camera.height, camera.width, 3), np.nan) if is_kerr else None
    counts, reasons = Counter(), Counter()
    maxima = dict(null_error=0.0, energy_drift=0.0, angular_momentum_drift=0.0)
    start = time.perf_counter()
    remaining = np.full(camera.height, camera.width)
    completed = 0
    results = (tracer.results(progress) if workers > 1 else
               ((row, col, momentum, tracer.ray(row, col, momentum))
                for row, col, momentum in camera.rays(metric)))
    from contextlib import closing
    with closing(results):
        for row, col, momentum, result in results:
            counts[result.status] += 1
            reasons[result.reason] += 1
            if is_numerical:
                info=result.numerical_diagnostics;levels=info['selected_levels'];level_counts.update(levels)
                numerical_failures.update(info['lookup_failures'])
                numerical_metric_calls+=info['evaluations'];numerical_stencils+=info['stencils']
                numerical_arrays['null_error'][row,col]=result.max_null_error
                numerical_arrays['finest_selected_level'][row,col]=max(levels,default=-1)
                numerical_arrays['selected_level_count'][row,col]=len(levels)
                numerical_arrays['capture_index'][row,col]=info['capture_index']
                numerical_arrays['time_slabs'][row,col]=info['slabs']
                numerical_arrays['temporal_stage_rejections'][row,col]=info['temporal_stage_rejections']
                numerical_arrays['lookup_stage_rejections'][row,col]=info['lookup_stage_rejections']
                numerical_arrays['constraint_diagnostics_complete'][row,col]=info['complete']
                numerical_reasons[row,col]=result.reason
            if is_binary:
                info=result.binary_diagnostics
                binary_arrays['capture_index'][row,col]=info['capture_index']
                binary_arrays['signature_margin'][row,col]=info['minimum_signature_margin']
                binary_arrays['constraint_diagnostics_complete'][row,col]=info['complete']
                binary_arrays['null_error'][row,col]=result.max_null_error
                binary_arrays['energy_drift'][row,col]=result.energy_drift
                binary_reasons[row,col]=result.reason
                if is_orbit_binary:
                    binary_arrays['energy_change'][row,col]=info.get('energy_change',np.nan)
                    binary_arrays['helical_error'][row,col]=info.get('helical_error',np.nan)
            if is_kerr:
                backend_records.record(row,col,result)
                for key, value in (("null_error", result.max_null_error), ("energy_drift", result.energy_drift),
                                   ("axial_momentum_drift", result.angular_momentum_drift),
                                   ("carter_drift", result.carter_drift)):
                    conservation[key][row, col] = value
                constants[row, col] = result.kerr_constants
            if result.status == "escaped":
                directions[row, col] = result.direction
                statuses[row, col] = 1
                image[row, col] = background.color(result.direction)
            elif result.status == "unresolved":
                statuses[row, col] = 2
                image[row, col] = (1.0, 0.0, 0.7)
            maxima["null_error"] = max(maxima["null_error"], result.max_null_error)
            for key in ("energy_drift", "angular_momentum_drift"):
                value = getattr(result, key)
                maxima[key] = max(maxima[key], value) if value is not None else None
            remaining[row] -= 1
            if remaining[row] == 0:
                completed += 1
                if progress:
                    progress(completed, camera.height)
    radius = np.linalg.norm(camera.position)
    diagnostics = {
        "method": "ode", "requested_method": method,
        "version": __version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "mass": getattr(metric, "mass", None), "camera": asdict(camera), "integration": asdict(integration),
        "counts": {k: counts[k] for k in ("captured", "escaped", "unresolved")},
        "termination_reasons": dict(reasons), "maxima": maxima,
        "diagnostic_sampling": "accepted steps and dense-output midpoints; null error = |g^ab p_a p_b| / p_t(initial)^2",
        "total_steps": tracer.steps, "rhs_evaluations": tracer.evaluations,
        "integrated_rays": tracer.integrated_rays,
        "rotational_symmetry_reuse": tracer.enabled,
        "elapsed_seconds": time.perf_counter() - start,
        "analytic_critical_impact_parameter": metric.critical_impact_parameter if type(metric) is Schwarzschild else None,
        "analytic_shadow_radius_degrees": float(np.rad2deg(metric.shadow_angle(radius))) if type(metric) is Schwarzschild and radius > 3*metric.mass else None,
        "background": background.description,
    }
    if workers > 1:
        diagnostics["execution"] = tracer.statistics
    if is_kerr:
        diagnostics.update(backend_records.metadata())
        diagnostics.update(spacetime={"type": "kerr", "spin": metric.spin, "a": metric.a,
                                      "axis": "+Z", "coordinates": "outgoing Cartesian Kerr-Schild"},
                           horizon_radius=metric.horizon_radius,
                           capture_radial_coordinate=integration.kerr_capture_fraction*metric.horizon_radius,
                           background_direction="Hamiltonian continuation to kerr_source_radius, normalized outgoing Cartesian tangent; residual bending O(M*b/R^2)",
                           conservation_normalization="|delta E|/|E0|, |delta Lz|/(M|E0|), |delta Q|/(M E0)^2; signed past-ray E,Lz,Q saved",
                           optimization_note="Kerr uses general ODEs; Schwarzschild quadrature and spherical rotational reuse disabled")
        diagnostics["maxima"].update(axial_momentum_drift=float(np.max(conservation["axial_momentum_drift"])),
                                      carter_drift=float(np.max(conservation["carter_drift"])))
        if integration.integrator == "carter":
            diagnostics.update(background_direction="Exact separated angular continuation in inverse radius to the same kerr_source_radius; final Cartesian tangent; residual finite-source bending O(M*b/R^2)",
                               optimization_note="Carter compact-radius exact-Kerr specialization; no Schwarzschild quadrature or spherical reuse; fallback provenance recorded separately")
    if is_binary:
        reason_names=sorted(reasons)
        binary_arrays['termination_code']=np.array([[reason_names.index(v) for v in row] for row in binary_reasons],dtype=np.uint16)
        diagnostics.update(spacetime=metric.description(),integrator='hamiltonian',
            provisional_capture_counts={str(i):int(np.count_nonzero(binary_arrays['capture_index']==i)) for i in (1,2)},
            capture_radii=[2*m*integration.binary_capture_fraction for m in metric.masses],
            escape_center=metric.barycenter.tolist(),
            background_direction='Normalized outgoing Cartesian tangent at finite escape_radius; residual bending requires radius convergence; no analytic tail',
            signature_guard=integration.binary_signature_margin,
            minimum_sampled_signature_margin=float(np.nanmin(binary_arrays['signature_margin'])),
            incomplete_diagnostics=int(np.count_nonzero(~binary_arrays['constraint_diagnostics_complete'])),
            termination_codes={str(i):v for i,v in enumerate(reason_names)},
            capture_index_codes={'0':'no provisional capture','1':'provisional sphere at center 1','2':'provisional sphere at center 2'},
            optimization_note='Generic Hamiltonian only; no Schwarzschild quadrature, rotational reuse or Carter integration')
        if is_orbit_binary:
            positions,velocities,accelerations=metric.body_states(camera.time)
            diagnostics.update(camera_time_body_positions=positions.tolist(),camera_time_body_velocities=velocities.tolist(),
                camera_time_body_accelerations=accelerations.tolist(),stationary_energy_conserved=False,
                capture_prescription='moving rest-radius ellipsoids, not computed horizons',
                capture_index_codes={'0':'no provisional capture','1':'provisional moving rest-radius ellipsoid 1','2':'provisional moving rest-radius ellipsoid 2'},
                maxima=diagnostics['maxima'] | dict(energy_change=float(np.nanmax(binary_arrays['energy_change'])) if np.isfinite(binary_arrays['energy_change']).any() else None,
                    helical_error=float(np.nanmax(binary_arrays['helical_error'])) if np.isfinite(binary_arrays['helical_error']).any() else None))
    if is_numerical:
        names=sorted(reasons)
        numerical_arrays['termination_code']=np.array([[names.index(v) for v in row] for row in numerical_reasons],dtype=np.uint16)
        diagnostics.update(spacetime=metric.description(),integrator='hamiltonian / guarded SciPy DOP853 temporal slabs',
            selected_level_ray_counts={str(k):v for k,v in sorted(level_counts.items())},
            numerical_metric_evaluations=numerical_metric_calls,numerical_stencils=numerical_stencils,
            lookup_failure_attempts=dict(numerical_failures),
            invalid_lookup_counts={k:v for k,v in reasons.items() if k.startswith('numerical_') and k not in ('numerical_supplied_capture','numerical_finite_background')},
            temporal_stage_rejections=int(numerical_arrays['temporal_stage_rejections'].sum()),
            lookup_stage_rejections=int(numerical_arrays['lookup_stage_rejections'].sum()),
            incomplete_diagnostics=int((~numerical_arrays['constraint_diagnostics_complete']).sum()),
            termination_codes={str(i):v for i,v in enumerate(names)},
            background_direction='Explicit finite direction-sphere only; normalized outgoing Cartesian tangent; no infinity or horizon inference',
            optimization_note='No Carter, Schwarzschild quadrature, spherical reuse, or native CarpetX runtime')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if is_numerical:
        (output/'numerical-dataset.toml').write_bytes((metric.data.root/'dataset.toml').read_bytes())
    Image.fromarray(np.uint8(np.round(np.clip(image, 0, 1) * 255))).save(output / "shadow.png")
    # Save computed directions for later background/exposure changes without ODEs.
    # Codes: 0=captured, 1=escaped, 2=unresolved. Non-escaping directions are NaN.
    np.savez_compressed(output / "rays.npz", directions=directions, status=statuses,
                        **conservation, **({"kerr_constants": constants} if is_kerr else {}),
                        **(backend_records.arrays() if is_kerr else {}), **binary_arrays, **numerical_arrays)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return diagnostics
