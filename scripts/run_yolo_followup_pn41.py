"""Wait for primary timings, repair paired probes, test calibrated/hybrid models."""
import json
from pathlib import Path
import subprocess
import time


def main():
    root=Path.cwd();out=root/'evaluations/yolo-comparison';deadline=time.monotonic()+7200
    while True:
        try:done=json.loads((out/'benchmark-manifest.json').read_text()).get('finished')
        except (OSError,ValueError):done=False
        if done:break
        if time.monotonic()>deadline:raise TimeoutError('Primary batch unfinished')
        time.sleep(5)
    models=[f'{m}_{s}' for m in ('yolov8n','yolo11n','yolo26n') for s in (320,480)]
    extras=[m+'_512' for m in ('yolo11n','yolo26n') if (out/'models'/f'{m}_512_int8_edgetpu.tflite').exists()]
    available=[m for m in models+extras if (out/'models'/f'{m}_int8_edgetpu.tflite').exists()]
    manifest={'started':time.time(),'paired':[],'extra_models':extras}
    def save():(out/'followup-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def module(name,*args,timeout=20000):
        return subprocess.run(['python3','-m','scripts.'+name,*args],timeout=timeout).returncode
    compose=['sudo','-n','docker','compose','--env-file','deploy/.env','-f','deploy/compose.yaml','-f','deploy/compose.pn41.yaml']
    running=subprocess.check_output(['sudo','-n','docker','inspect','--format','{{.State.Running}}','indoorcam-analytics-1'],text=True).strip()=='true'
    subprocess.run(compose+['stop','analytics'],check=True);save()
    try:
        viable=[]
        if extras:
            module('run_yolo_smoke_pn41','--models',*extras)
            p=max(out.glob('smoke-*.json'),key=lambda p:p.stat().st_mtime);rows=json.loads(p.read_text())
            manifest['extra_screen']=rows
            viable=[m for m in extras if len([r for r in rows if r['model']==m])==2 and
                    all(r['returncode']==0 and r.get('pipeline_fps',0)>=2 for r in rows if r['model']==m)]
            save()
        for model in available:
            name=f'indoorcam-yolo-paired-{model}'
            cmd=['sudo','-n','docker','run','--rm','--name',name,'--network','none','--user','0',
                 '--device','/dev/apex_0','--device','/dev/apex_1','-v',f'{root}:/work','-w','/work',
                 '--entrypoint','python','indoorcam:pn41','-m','scripts.probe_yolo_pose','--models',model,
                 '--backend','coral','--model-root','evaluations/yolo-comparison/models',
                 '--images-root','evaluations/pose-comparison/review-frames',
                 '--out',f'evaluations/yolo-comparison/paired/{model}.json']
            try:
                with (out/f'paired-final-{model}.log').open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['sudo','-n','docker','rm','-f',name],capture_output=True);rc=124
            manifest['paired'].append({'model':model,'returncode':rc});save();print('PAIRED',manifest['paired'][-1],flush=True)
        manifest['full_models']=['yolov8n_480_cal010','yolov8n_480_thunder']+viable;save()
        manifest['full_returncode']=module('run_pose_comparison_pn41','--models',*manifest['full_models'],
                '--model-root','evaluations/yolo-comparison','--output','evaluations/yolo-comparison/followup-results')
    finally:
        if running:subprocess.run(compose+['start','analytics'],check=True)
        manifest['finished']=time.time();save()


if __name__=='__main__':main()
