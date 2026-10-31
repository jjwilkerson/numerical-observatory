"""Deterministic per-subray provenance for the optional Carter backend."""
from collections import Counter
import numpy as np


class BackendRecords:
    def __init__(self,shape,settings):
        self.settings=settings;self.enabled=settings.integrator=='carter'
        self.counts=Counter();self.failures=Counter();self.fallback_count=0;self.accepted=0
        if self.enabled:
            self.backend=np.zeros(shape,dtype=np.uint8)
            self.reasons=np.empty(shape,dtype=object);self.reasons.fill('')
            self.radial=np.full(shape,np.nan);self.polar=np.full(shape,np.nan)
            self.complete=np.zeros(shape,dtype=bool)

    def record(self,row,col,result):
        self.counts[result.backend]+=1
        info=result.carter_diagnostics or {}
        failure=result.fallback_reason or (info.get('reason') if info.get('failed') else None)
        if not failure and result.status=='unresolved' and self.enabled:
            failure=result.reason
        if failure:self.failures[failure]+=1
        self.fallback_count+=int(result.fallback_reason is not None)
        self.accepted+=int(result.backend=='carter' and result.status!='unresolved' and not failure)
        if self.enabled:
            self.backend[row,col]=int(result.backend=='carter')
            self.reasons[row,col]=failure or ''
            self.radial[row,col]=info.get('radial_first_integral_error',np.nan)
            self.polar[row,col]=info.get('polar_first_integral_error',np.nan)
            self.complete[row,col]=info.get('constraint_diagnostics_complete',not info.get('failed',False))

    def arrays(self):
        if not self.enabled:return {}
        # Sorting avoids worker-completion order changing numerical reason IDs.
        reasons=['']+sorted(self.failures)
        codes=np.zeros(self.backend.shape,dtype=np.uint16)
        for code,reason in enumerate(reasons[1:],1): codes[self.reasons==reason]=code
        return dict(backend=self.backend,backend_failure_code=codes,
                    carter_radial_error=self.radial,carter_polar_error=self.polar,
                    carter_constraints_complete=self.complete)

    def metadata(self):
        reasons=['']+sorted(self.failures)
        return dict(integrator=self.settings.integrator,carter_fallback=self.settings.carter_fallback,
                    final_backend_counts=dict(sorted(self.counts.items())),
                    carter_accepted_rays=self.accepted,carter_failure_reasons=dict(sorted(self.failures.items())),
                    fallback_count=self.fallback_count,
                    backend_codes={'0':'hamiltonian','1':'carter'},
                    backend_failure_codes={str(i):s for i,s in enumerate(reasons)},
                    carter_failure_sampling='Incomplete constraint diagnostics are explicitly marked; early failures may contain only the initial-state constraint measurement.')
