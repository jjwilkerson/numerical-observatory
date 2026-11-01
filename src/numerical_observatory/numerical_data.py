"""GMDS v1 metadata and read-only native ADM payloads (no CarpetX runtime)."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,tomllib,math
import numpy as np

FIELDS='alp betax betay betaz gxx gxy gxz gyy gyz gzz'
COMPONENT=np.array([[4,5,6],[5,7,8],[6,8,9]])


class NumericalLookupError(ValueError):
    def __init__(self,code,message):
        self.code=code;super().__init__(f'{code}: {message}')


def require(condition,message):
    if not condition:raise NumericalLookupError('numerical_invalid_metadata',message)


def validate_fields(fields,context='ADM interpolant'):
    a=np.asarray(fields)
    if not np.isfinite(a).all():raise NumericalLookupError('numerical_nonfinite_fields',context)
    if np.any(a[...,0]<=0):raise NumericalLookupError('numerical_invalid_lapse',context)
    try:np.linalg.cholesky(a[...,COMPONENT])
    except np.linalg.LinAlgError as exc:raise NumericalLookupError('numerical_non_spd_gamma',context) from exc


@dataclass(frozen=True)
class Patch:
    level:int
    shape:tuple
    origin:tuple
    spacing:tuple
    centering:str
    file:str

    @property
    def size(self):return math.prod(self.shape)*80
    @property
    def upper(self):return np.asarray(self.origin)+(np.asarray(self.shape)-1)*self.spacing
    @property
    def geometry(self):return self.level,self.shape,self.origin,self.spacing,self.centering


class GMDSDataset:
    def __init__(self,directory,cache_bytes=1024**3):
        self.root=Path(directory).resolve();self.cache_bytes=int(cache_bytes)
        require(self.cache_bytes>0,'cache budget must be positive')
        manifest=self.root/'dataset.toml'
        try:
            raw=manifest.read_bytes();self.metadata=tomllib.loads(raw.decode('utf-8'))
        except (OSError,ValueError) as exc:raise NumericalLookupError('numerical_invalid_metadata',str(exc)) from exc
        self.manifest_sha256=hashlib.sha256(raw).hexdigest();m=self.metadata
        for key,value in [('schema_version',1),('coordinates','cartesian_txyz'),('signature','-+++'),('symmetry','none'),('fields',FIELDS)]:
            require(m.get(key)==value,f'unsupported {key}: expected {value}')
        require(isinstance(m.get('identifier'),str) and bool(m['identifier']),'identifier required')
        require(isinstance(m.get('units'),str) and bool(m['units']),'units required')
        snapshots=m.get('snapshots',[]);require(len(snapshots)>=2,'at least two saved snapshots required; no implicit frozen extension')
        self.times=np.asarray([s.get('time',np.nan) for s in snapshots],dtype=float)
        require(np.isfinite(self.times).all() and np.all(np.diff(self.times)>0),'finite strictly increasing times required')
        self.snapshots=[]
        for snapshot in snapshots:
            patches=[]
            for p in snapshot.get('patches',[]):
                try:q=Patch(p['level'],tuple(p['shape']),tuple(p['origin']),tuple(p['spacing']),p['centering'],p['file'])
                except (KeyError,TypeError) as exc:raise NumericalLookupError('numerical_invalid_metadata','incomplete patch metadata') from exc
                require(isinstance(q.level,int) and q.level>=0,'invalid level')
                require(len(q.shape)==len(q.origin)==len(q.spacing)==3,'expected xyz triples')
                require(all(isinstance(v,int) and v>=2 for v in q.shape),'shape entries must be integers >=2')
                require(np.isfinite(q.origin).all() and np.isfinite(q.spacing).all() and min(q.spacing)>0 and np.isfinite(q.upper).all(),'invalid patch extent')
                require(q.centering in ('vertex','cell'),'unsupported centering')
                path=Path(q.file)
                require(not path.is_absolute() and '..' not in path.parts and bool(q.file),'unsafe patch path')
                resolved=(self.root/path).resolve()
                require(resolved.is_relative_to(self.root),'patch path leaves dataset')
                require(resolved.is_file() and resolved.stat().st_size==q.size,f'payload byte-size mismatch: {q.file}')
                for old in patches:
                    if old.level!=q.level:continue
                    require(np.all(abs(np.asarray(old.spacing)-q.spacing)<=1e-12*np.asarray(q.spacing)),'inconsistent level spacing')
                    offset=(np.asarray(q.origin)-old.origin)/q.spacing
                    require(np.all(abs(offset-np.rint(offset))<1e-8),'unaligned same-level patches')
                patches.append(q)
            require(bool(patches),'snapshot has no patches')
            self.snapshots.append(patches)
        original=[p.geometry for p in self.snapshots[0]]
        require(all([p.geometry for p in ps]==original for ps in self.snapshots),'Phase 1 requires fixed ordered spatial hierarchy; dynamic regridding unsupported')
        require(all(sum(p.size for p in self.snapshots[i])+sum(p.size for p in self.snapshots[i+1])<=self.cache_bytes for i in range(len(self.times)-1)),'active snapshot pair exceeds cache budget')
        self._cache={};self.loaded_bytes=0;self.loads=0

    def activate(self,lo,hi):
        for key in list(self._cache):
            if key not in (lo,hi):del self._cache[key]
        for index in (lo,hi):
            if index in self._cache:continue
            mapped=[]
            for patch in self.snapshots[index]:
                a=np.memmap(self.root/patch.file,dtype='<f8',mode='r',shape=(patch.shape[2],patch.shape[1],patch.shape[0],10))
                flat=a.reshape(-1,10)
                for start in range(0,len(flat),32768):validate_fields(flat[start:start+32768],f'native snapshot {index}, level {patch.level}, {patch.file}')
                mapped.append(a)
            self._cache[index]=mapped;self.loads+=1
        self.loaded_bytes=sum(p.size for i in self._cache for p in self.snapshots[i])

    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']={};state['loaded_bytes']=0;state['loads']=0
        return state  # never pickle mapped payload into spawn workers

    def identity(self):
        return dict(identifier=self.metadata['identifier'],directory=str(self.root),format='GMDS v1, converted physical CarpetX ADM',
                    manifest_sha256=self.manifest_sha256,fields=FIELDS.split(),units=self.metadata['units'],
                    time_coverage=[float(self.times[0]),float(self.times[-1])],snapshots=len(self.times),
                    snapshot_times=self.times.tolist(),native_regions=self.metadata.get('native_regions',[]),
                    fixed_geometry=True,raw_bytes_per_snapshot=sum(p.size for p in self.snapshots[0]),cache_budget=self.cache_bytes,
                    source_series=self.metadata.get('source_series'),source_time_unit_si=self.metadata.get('source_time_unit_si'),
                    patches=[dict(level=p.level,shape=p.shape,origin=p.origin,spacing=p.spacing,centering=p.centering,file=p.file) for p in self.snapshots[0]],
                    payload_policy='immutable read-only mmap; active snapshots globally physically validated; payload never pickled')
