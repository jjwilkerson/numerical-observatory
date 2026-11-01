"""Optional external C++ validation process; no C++ dependency in rendering."""
import json,subprocess
from pathlib import Path
import numpy as np


class CppOracle:
    def __init__(self,config):
        self.config=json.loads(Path(config).read_text())
        self.command=self.config['command']
        if not isinstance(self.command,list) or not all(isinstance(s,str) for s in self.command):raise ValueError('oracle command must be an argument list')
    def path(self,path):
        path=Path(path).resolve()
        mapping=self.config.get('path_map')
        return str(path) if mapping is None else str(mapping['remote']).rstrip('/')+'/'+path.relative_to(Path(mapping['local']).resolve()).as_posix()
    def run(self,dataset,mode,action,rows):
        body='\n'.join(' '.join(format(float(v),'.17g') for v in row) for row in rows)+'\n'
        result=subprocess.run(self.command+[self.path(dataset),mode,action],input=body,text=True,capture_output=True,timeout=180,check=False)
        if result.returncode:raise RuntimeError('C++ oracle failed: '+result.stderr)
        return result.stdout.splitlines()
    def probes(self,dataset,mode,events):
        values=[];levels=[]
        for line in self.run(dataset,mode,'probe',events):
            if not line.startswith('OK '):values.append(None);levels.append([]);continue
            tokens=line.split()[1:];values.append(np.array(list(map(float,tokens[:80]))).reshape(5,4,4))
            levels.append(sorted(set(int(s[1:]) for s in tokens[80:])))
        if len(values)!=len(events):raise RuntimeError('C++ probe row count mismatch')
        return values,levels
