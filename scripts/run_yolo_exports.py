"""Sequential isolated export processes, with resumable per-variant manifests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants',nargs='+',default=[f'{m}_{s}' for m in
        ('yolov8n','yolo11n','yolo26n') for s in (320,480)])
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--batch-name',default='export-batch')
    args=parser.parse_args();root=Path('artifacts/yolo-comparison');manifest={'started':time.time(),'runs':[]}
    for variant in args.variants:
        if (root/'exports'/variant/'export.json').exists() and not args.force:continue
        model,size=variant.rsplit('_',1)
        with (root/f'export-{variant}.log').open('w') as log:
            print('START',variant,flush=True);tick=time.time()
            try:
                p=subprocess.run([sys.executable,'-m','scripts.export_yolo_pose','--model',model+'-pose',
                    '--size',size],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                row={'variant':variant,'returncode':p.returncode,'elapsed_s':time.time()-tick}
            except subprocess.TimeoutExpired:
                row={'variant':variant,'error':'timeout_1200s'}
        manifest['runs'].append(row);(root/(args.batch_name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        print('END',row,flush=True)
    manifest['finished']=time.time();(root/(args.batch_name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
