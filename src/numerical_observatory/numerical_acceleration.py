"""Thin optional-kernel adapter; data loading and certification remain Python.

Mapping views are local to one provider/worker and are never pickled. No global
scratch, payload copies, threading, or approximate ray/metric caches are used.
"""
import numpy as np
from .numerical_data import NumericalLookupError


class CompiledEvaluator:
    def __init__(self,metric):
        try:
            from numba.typed import List
            from . import numerical_kernels as kernels
        except ImportError as exc:
            raise ValueError('backend="numba" requires the optional numerical-observatory[acceleration] extra') from exc
        self.List=List;self.kernels=kernels;self.mappings={}
        self.active_pair=None;self.pair_buffers=None;self.pair_sources=None
        geometry=metric._geometry
        from .numerical_metric import MODES
        self.args=(geometry.levels,geometry.origins,geometry.spacings,geometry.shapes,
                   np.array(metric.levels,dtype=np.int64),
                   np.array([metric.data.snapshots[0].index(metric.anchors[l]) for l in metric.levels],dtype=np.int64),
                   np.array(sorted(range(len(geometry.levels)),key=lambda p:-geometry.levels[p]),dtype=np.int64),
                   MODES.index(metric.mode),
                   np.array([[b.outer_lower,b.inner_lower,b.inner_upper,b.outer_upper] for b in metric.bands],dtype=float).reshape(-1,4,3),
                   np.array([[b.coarse_level,b.fine_level] for b in metric.bands],dtype=np.int64).reshape(-1,2),
                   np.array([metric.transition[k] for k in ('axis','coarse_level','fine_level','lower','upper','fine_on_lower')],dtype=float) if metric.transition else np.zeros(6))

    def buffers(self,m,si):
        # Keep only the active pair, including when tracing moves backwards.
        self.mappings={i:v for i,v in self.mappings.items() if i in m.data._cache and v[0] is m.data._cache[i]}
        if si not in self.mappings:
            arrays=self.List()
            for mapped in m.data._cache[si]:arrays.append(np.asarray(mapped))
            self.mappings[si]=(m.data._cache[si],arrays)
        return self.mappings[si][1]

    def prepare(self,m,event,slab):
        x=np.asarray(event,dtype=float)
        if x.shape!=(4,) or not np.isfinite(x).all():raise NumericalLookupError('numerical_invalid_event','expected finite txyz')
        lo,hi=m.interval(x[0]) if slab is None else slab
        a=m.data.times[lo];b=m.data.times[hi]
        if x[0]<a or x[0]>b:raise NumericalLookupError('numerical_temporal_boundary','outside selected closed slab')
        if (self.active_pair!=(lo,hi) or m.data._cache.get(lo) is not self.pair_sources[0]
                or m.data._cache.get(hi) is not self.pair_sources[1]):
            # Drop old views BEFORE mapping/validating the next pair. Validate
            # and rebuild exactly when maps change; never cache field values.
            self.pair_buffers=self.pair_sources=None;self.mappings.clear();self.active_pair=None
            m.data.activate(lo,hi)
            self.pair_buffers=(self.buffers(m,lo),self.buffers(m,hi))
            self.pair_sources=(m.data._cache[lo],m.data._cache[hi]);self.active_pair=(lo,hi)
        return (*self.pair_buffers,*self.args,x[1:],(x[0]-a)/(b-a),b-a)

    @staticmethod
    def record(m,stats):
        m.stencils+=int(stats[0])
        for index in np.flatnonzero(stats[1:]):
            level=m.levels[index];m.last_levels.add(level);m.selected_levels.add(level)

    def evaluate(self,m,event,slab):
        f,df,stats,error=self.kernels.adm(*self.prepare(m,event,slab))
        from .numerical_metric import reconstruct
        if error:
            # Original evaluation supplies the exact validity decision/reason and
            # counters. Kernel attempts are deliberately not counted twice.
            m.backend_fallbacks+=1
            return reconstruct(*m._adm_reference(event,slab))
        # Keep the small 3+1 algebra on the reference LAPACK/NumPy path. The
        # compiled reconstruction prototype differs by a few ulps, which can
        # noticeably change adaptive step histories on strongly curved grids.
        self.record(m,stats)
        from .numerical_numpy import reconstruct as reconstruct_numpy
        return reconstruct_numpy(f,df)

    def adm(self,m,event,slab):
        f,df,stats,error=self.kernels.adm(*self.prepare(m,event,slab))
        if error:
            m.backend_fallbacks+=1;return m._adm_reference(event,slab)
        self.record(m,stats);return f,df

    def spatial(self,m,si,x):
        stats=np.zeros(len(m.levels)+1,dtype=np.int64)
        result,error=self.kernels.spatial(self.buffers(m,si),*self.args,np.asarray(x,dtype=float),stats)
        if error:
            m.backend_fallbacks+=1;return m._spatial_reference(si,x)
        self.record(m,stats);return result
