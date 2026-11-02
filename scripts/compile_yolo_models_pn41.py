"""Compile exported INT8 models in a resource-limited, network-disabled container."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--head-cuts',action='store_true')
    args=p.parse_args();root=Path('evaluations/yolo-comparison').resolve()
    mode='headcut' if args.head_cuts else 'tfgraph'
    out=root/'compiled'/mode;out.mkdir(parents=True,exist_ok=True)
    manifest={'started':time.time(),'runs':[],'flags':['show_operations','explicit_head_cuts'] if args.head_cuts else ['show_operations','search_delegate','delegate_search_step=30']}
    bundle='/eval/tools/usr/bin/edgetpu_compiler_bin'
    for name in args.models:
        source=root/'models'/f'{name}_int8.tflite'
        log=root/f'compile-{mode}-{name}.log'
        cmd=['sudo','-n','docker','run','--rm','--name',f'indoorcam-yolo-compile-{name}',
             '--network','none','--memory','768m','--memory-swap','768m','--cpus','2',
             '-v',f'{root}:/eval','-w','/eval','--entrypoint',bundle+'/ld-linux-x86-64.so.2',
             'indoorcam:pn41','--library-path',bundle,bundle+'/edgetpu_compiler',
             '--show_operations','--out_dir',f'/eval/compiled/{mode}','--search_delegate',
             '--delegate_search_step','30','--timeout_sec','180',f'/eval/models/{source.name}']
        if args.head_cuts:
            names=json.loads((root/'head-cuts.json').read_text())[name]
            cmd.remove('--search_delegate')
            index=cmd.index('--delegate_search_step');del cmd[index:index+2]
            cmd[-1:-1]=['--intermediate_tensors',','.join(names)]
        print('START',name,flush=True);start=time.time()
        try:
            with log.open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=420).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['sudo','-n','docker','rm','-f',f'indoorcam-yolo-compile-{name}'],capture_output=True)
            rc=124
        text=log.read_text();row={'model':name,'returncode':rc,'elapsed_s':time.time()-start,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        for kind,pattern in [('tpu_ops',r'Number of operations that will run on Edge TPU: (\d+)'),
                             ('cpu_ops',r'Number of operations that will run on CPU: (\d+)')]:
            matches=re.findall(pattern,text)
            if matches:row[kind]=int(matches[-1])
        compiled=out/f'{name}_int8_edgetpu.tflite'
        if rc==0 and compiled.exists():
            shutil.copy2(compiled,root/'models'/compiled.name)
            row['compiled_sha256']=hashlib.sha256(compiled.read_bytes()).hexdigest()
        manifest['runs'].append(row);(root/f'compile-{mode}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('END',row,flush=True)
    manifest['finished']=time.time();(root/f'compile-{mode}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
