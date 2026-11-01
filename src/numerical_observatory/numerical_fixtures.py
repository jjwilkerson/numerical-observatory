"""Small explicit synthetic GMDS writers for numerical/cross-language validation."""
from pathlib import Path
import json
import numpy as np
from .numerical_data import COMPONENT,validate_fields,FIELDS


def adm_from_metric(metric,event):
    inv=metric.inverse(event);cov=metric.covariant(event);alpha=np.sqrt(-1/inv[0,0])
    return np.r_[alpha,inv[0,1:]*alpha**2,cov[1,1],cov[1,2],cov[1,3],cov[2,2],cov[2,3],cov[3,3]]


def write_dataset(directory,times,patches,sampler,native=True,stationary=False,identifier='synthetic-observatory'):
    root=Path(directory)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    text=['schema_version = 1',f'identifier = {json.dumps(identifier)}','coordinates = "cartesian_txyz"',
          'signature = "-+++"','units = "geometrized code units; no SI conversion"','symmetry = "none"',
          f'fields = "{FIELDS}"',f'synthetic_temporal_model = "{"known_stationary_analytic_samples" if stationary else "sampled_time_dependent_fields"}"']
    for si,t in enumerate(times):
        text+=['','[[snapshots]]',f'time = {float(t)!r}']
        for pi,p in enumerate(patches):
            name=f'patch-{pi if stationary else si*len(patches)+pi}.bin'
            if not stationary or si==0:
                nx,ny,nz=p['shape'];origin=np.array(p['origin']);h=np.array(p['spacing'])
                data=np.empty((nz,ny,nx,10))
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):data[k,j,i]=sampler(float(t),origin+h*[i,j,k],p['level'])
                validate_fields(data,'generated exact/manufactured samples');data.astype('<f8').tofile(root/name)
            text+=['[[snapshots.patches]]',f'level = {p["level"]}',f'centering = "{p.get("centering","vertex")}"',
                   f'shape = {list(map(int,p["shape"]))}',f'origin = {list(map(float,p["origin"]))}',
                   f'spacing = {list(map(float,p["spacing"]))}',f'file = "{name}"']
    if native:
        for level in sorted(set(p['level'] for p in patches)):
            ps=[p for p in patches if p['level']==level]
            lower=np.min([p['origin'] for p in ps],axis=0)
            upper=np.max([np.array(p['origin'])+(np.array(p['shape'])-1)*p['spacing'] for p in ps],axis=0)
            text+=['','[[native_regions]]',f'level = {level}','provenance = "analytic_sampling"',
                   f'lower = {lower.tolist()}',f'upper = {upper.tolist()}']
    (root/'dataset.toml').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return root


def flat_fields(time,xyz,level):return np.array([1,0,0,0,1,0,0,1,0,1.])


def manufactured_fields(time,xyz,level):
    x,y,z=xyz
    return np.array([1.5+.03*time+.02*x+.01*y-.015*z+.02*level,
                     .01+.001*x+.002*time,-.02+.002*y,.005+.001*z,
                     1.2+.01*x+.03*level,.003*y,.002*z,1.3+.01*y,.002*x,1.4+.01*z])


def nested_boxes(levels=3,spacing=1.,center=(10,0,0)):
    boxes=[]
    for level in range(levels):
        h=spacing/2**level;half=8/2**level;n=round(2*half/h)+1
        boxes.append(dict(level=level,shape=[n]*3,origin=(np.array(center)-half).tolist(),spacing=[h]*3))
    return boxes
