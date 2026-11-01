"""Optional native CarpetX ADMBaseX openPMD/ADIOS2 -> portable GMDS v1.

Faithful chunk/axis/centering conversion, not a general openPMD importer.
Only this preprocessing command needs the optional openpmd-api dependency.
"""
import argparse,json,re,tomllib
from pathlib import Path
import numpy as np
from .amr_geometry import uncovered
from .numerical_data import FIELDS,validate_fields


def convert(series_path,destination,identifier,crop=None):
    try:import openpmd_api as io
    except ImportError as exc:raise RuntimeError('native BP5 conversion needs pip install "numerical-observatory[carpetx]"; GMDS evaluation does not') from exc
    if not io.variants.get('adios2',False):raise RuntimeError('installed openPMD reader lacks ADIOS2 support')
    path=Path(series_path)
    if path.suffix.lower() not in ('.bp','.bp5'):raise ValueError('Phase 1 supports the reference ADIOS2 BP/BP5 format only')
    dest=Path(destination)
    if dest.exists():raise FileExistsError(dest)
    if not identifier or not re.fullmatch(r'[A-Za-z0-9_. -]+',identifier):raise ValueError('invalid identifier')
    if crop is not None:
        crop=np.asarray(crop,dtype=float).reshape(3,2)
        if not np.isfinite(crop).all() or not np.all(crop[:,0]<crop[:,1]):raise ValueError('invalid xyz crop bounds')
    series=io.Series(str(path.resolve()),io.Access.read_only)
    dest.mkdir(parents=True);manifest=dest/'dataset.toml.incomplete'
    text=['schema_version = 1',f'identifier = {json.dumps(identifier)}','coordinates = "cartesian_txyz"',
          'signature = "-+++"','units = "CarpetX code units, c=G=1"','symmetry = "none"',f'fields = "{FIELDS}"',
          f'source_series = {json.dumps(str(path.resolve()))}',f'conversion_library = "openPMD-api {io.__version__}"']
    manifest.write_text('\n'.join(text)+'\n')
    previous=-np.inf;time_unit=length_unit=None;native_original=None;geometry_original=None;patch_count=0
    try:
        count=0
        for iteration_id,it in series.iterations.items():
            time=float(it.time)
            if not np.isfinite(time) or time<=previous:raise ValueError('unordered/nonfinite source times')
            previous=time;count+=1
            if time_unit is None:
                time_unit=float(it.time_unit_SI);text.append(f'source_time_unit_si = {time_unit!r}')
            if not np.isfinite(time_unit) or time_unit<=0 or time_unit!=it.time_unit_SI:raise ValueError('invalid/changing source time unit')
            text+=['','[[snapshots]]',f'time = {time!r}',f'source_iteration = {iteration_id}']
            native={};geometry=[];written=0
            for name,mesh in it.meshes.items():
                match=re.fullmatch(r'admbasex_lapse_patch([0-9]+)_lev([0-9]+)',name)
                if not match:continue
                if int(match[1])!=0:raise ValueError('only Cartesian coordinate patch 0 supported')
                level=int(match[2]);suffix=name[len('admbasex_lapse'):];records=[];baseline=None
                for field,var in enumerate(FIELDS.split()):
                    group='lapse' if field==0 else 'shift' if field<4 else 'metric'
                    try:m=it.meshes['admbasex_'+group+suffix];rc=m['admbasex_'+var]
                    except (KeyError,IndexError) as exc:raise ValueError(f'missing required ADMBaseX {var}') from exc
                    if list(m.axis_labels)!=['z','y','x'] or m.geometry!=io.Geometry.cartesian or m.time_offset!=0:
                        raise ValueError('unsupported axes, geometry or mesh time offset')
                    if rc.unit_SI!=1.:raise ValueError('nontrivial ADM component unitSI unsupported')
                    if np.dtype(rc.dtype)!=np.dtype('float64'):raise ValueError('reference native ADM precision must be float64')
                    unit=float(m.grid_unit_SI)
                    if length_unit is None:length_unit=unit
                    if not np.isfinite(unit) or unit<=0 or unit!=length_unit:raise ValueError('invalid/changing grid unit')
                    spacing=np.array(m.grid_spacing,dtype=float)[::-1];origin=np.array(m.grid_global_offset,dtype=float)[::-1]
                    position=np.array(rc.position,dtype=float)[::-1]
                    if spacing.shape!=(3,) or origin.shape!=(3,) or position.shape!=(3,) or not np.isfinite([spacing,origin,position]).all() or min(spacing)<=0:
                        raise ValueError('invalid Cartesian grid geometry')
                    chunks=[(tuple(map(int,c.offset)),tuple(map(int,c.extent))) for c in rc.available_chunks()]
                    current=(tuple(spacing),tuple(origin),tuple(position),chunks)
                    if baseline is None:baseline=current
                    elif current!=baseline:raise ValueError('inconsistent ADM groups, centering or valid chunk boxes')
                    records.append(rc)
                if not (np.all(position==0) or np.all(position==.5)):raise ValueError('mixed centering unsupported')
                vertex=np.all(position==0);chunks=baseline[3]
                if not chunks:raise ValueError('empty source level')
                boxes=[(np.array(offset[::-1]),np.array(offset[::-1])+extent[::-1]) for offset,extent in chunks]
                lower=np.min([b[0] for b in boxes],axis=0);upper=np.max([b[1] for b in boxes],axis=0)
                rectangular=not uncovered((lower,upper),boxes)
                native[level]=(tuple(origin+(lower+position)*spacing),tuple(origin+(upper-1+position)*spacing)) if rectangular else None
                geometry.append((level,baseline))
                for offset,extent in chunks:
                    start=np.array(offset[::-1]);end=start+extent[::-1]
                    if crop is not None:
                        first=origin+position*spacing
                        start=np.maximum(start,np.ceil((crop[:,0]-first)/spacing-1e-10).astype(int))
                        end=np.minimum(end,np.floor((crop[:,1]-first)/spacing+1e-10).astype(int)+1)
                    shape=end-start
                    if np.any(shape<2):continue
                    arrays=[r.load_chunk(start[::-1].tolist(),shape[::-1].tolist()) for r in records]
                    series.flush()
                    values=np.stack(arrays,axis=-1);validate_fields(values,f'source iteration {iteration_id}, level {level}, chunk {offset}')
                    file=f'patch-{patch_count}.bin';patch_count+=1;values.astype('<f8').tofile(dest/file)
                    sample_origin=origin+(start+position)*spacing
                    text+=['[[snapshots.patches]]',f'source_length_unit_si = {unit!r}',f'level = {level}',
                           f'centering = "{"vertex" if vertex else "cell"}"',f'shape = {shape.tolist()}',
                           f'origin = {sample_origin.tolist()}',f'spacing = {spacing.tolist()}',f'file = "{file}"']
                    written+=1
            if not written:raise ValueError('no complete retained physical ADM patches')
            if native_original is None:native_original=native;geometry_original=geometry
            elif native!=native_original or geometry!=geometry_original:raise ValueError('dynamic regridding/changing native chunks unsupported in Phase 1')
            it.close()
        if count<2:raise ValueError('at least two source snapshots required')
        if all(v is not None for v in native_original.values()):
            for level,(lo,hi) in sorted(native_original.items()):
                text+=['','[[native_regions]]',f'level = {level}','provenance = "carpetx_valid_chunks"',f'lower = {list(map(float,lo))}',f'upper = {list(map(float,hi))}']
        content='\n'.join(text)+'\n';tomllib.loads(content)
        manifest.write_text(content,encoding='utf-8');manifest.rename(dest/'dataset.toml')
    finally:series.close()
    return dict(snapshots=count,patches=patch_count,directory=str(dest.resolve()),format='GMDS v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('series');parser.add_argument('destination',type=Path);parser.add_argument('identifier')
    parser.add_argument('--crop',nargs=6,type=float,metavar=('XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'))
    args=parser.parse_args();print(json.dumps(convert(args.series,args.destination,args.identifier,args.crop),indent=2))


if __name__=='__main__':main()
