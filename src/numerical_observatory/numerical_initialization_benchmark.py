"""Separate metadata/certification, mmap opening and native physical validation."""
import argparse,json
from pathlib import Path
import numpy as np
from .numerical_data import GMDSDataset,validate_fields
from .numerical_metric import NumericalMetric
from .numerical_benchmark import timed,machine,memory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--time',type=float,default=4.)
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    data=GMDSDataset(a.dataset);hi=int(np.clip(np.searchsorted(data.times,a.time,side='right'),1,len(data.times)-1));lo=hi-1
    def open_maps():
        return [np.memmap(data.root/q.file,dtype='<f8',mode='r',shape=(q.shape[2],q.shape[1],q.shape[0],10)) for i in (lo,hi) for q in data.snapshots[i]]
    maps=open_maps()
    def validation():
        for array in maps:
            values=array.reshape(-1,10)
            for start in range(0,len(values),32768):validate_fields(values[start:start+32768])
    def activate():
        dataset=GMDSDataset(a.dataset);dataset.activate(lo,hi)
    result=dict(machine=machine(),dataset=data.identity(),snapshot_pair=[lo,hi],repeats=a.repeats,
        metadata=timed(lambda:GMDSDataset(a.dataset),a.repeats),
        hierarchy_constructor=timed(lambda:NumericalMetric(a.dataset,'cubic_amr_c1_hierarchy'),a.repeats),
        mmap_open_no_touch=timed(open_maps,a.repeats),native_validation_warm_pages=timed(validation,a.repeats),
        metadata_map_validate=timed(activate,a.repeats),process_memory=memory(),
        notes='native validation uses the unchanged loader check and chunk size; OS page cache is not evicted; mmap opening is lazy and is not disk bandwidth')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
