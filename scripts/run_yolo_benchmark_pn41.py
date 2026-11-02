"""Compile, screen, probe all YOLOs, then fully replay viable configurations.

Screen: >=2 fps in both 16-frame clips (three warmups excluded). This is a
screening criterion, not a published full-video throughput measurement.
"""
import json
from pathlib import Path
import shutil
import subprocess
import time


MODELS=[f'{m}_{s}' for m in ('yolov8n','yolo11n','yolo26n') for s in (320,480)]


def main():
    root=Path.cwd();out=root/'evaluations/yolo-comparison';manifest={'started':time.time(),'screen_min_fps':2,'models':MODELS}
    def save():
        (out/'benchmark-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def module(name,*args,timeout=7200):
        return subprocess.run(['python3','-m','scripts.'+name,*args],timeout=timeout).returncode
    save()
    # No devices are exposed to compiler containers; production can keep running.
    module('compile_yolo_models_pn41','--head-cuts','--models',*MODELS)
    cut=json.loads((out/'compile-headcut-manifest.json').read_text())
    failed=[r['model'] for r in cut['runs'] if r['returncode']!=0]
    if failed:module('compile_yolo_models_pn41','--models',*failed)
    available=[m for m in MODELS if (out/'models'/f'{m}_int8_edgetpu.tflite').exists()]
    manifest['compiled_available']=available;save()
    compose=['sudo','-n','docker','compose','--env-file','deploy/.env','-f','deploy/compose.yaml','-f','deploy/compose.pn41.yaml']
    running=subprocess.check_output(['sudo','-n','docker','inspect','--format','{{.State.Running}}','indoorcam-analytics-1'],text=True).strip()=='true'
    subprocess.run(compose+['stop','analytics'],check=True)
    try:
        if available:module('run_yolo_smoke_pn41','--models',*available)
        smokes=sorted(out.glob('smoke-*.json'),key=lambda p:p.stat().st_mtime)
        rows=json.loads(smokes[-1].read_text()) if smokes else []
        manifest['screen_rows']=rows
        viable=[m for m in available if len([r for r in rows if r['model']==m])==2 and
                all(r['returncode']==0 and r.get('pipeline_fps',0)>=2 for r in rows if r['model']==m)]
        manifest['full_video_yolo_models']=viable;save()
        # Pair exact JPEG images with local FP32/INT8 probes. Never publish HA state.
        paired=[]
        for model in available:
            name=f'indoorcam-yolo-probe-{model}'
            cmd=['sudo','-n','docker','run','--rm','--name',name,'--network','none','--user','0',
                 '--device','/dev/apex_0','--device','/dev/apex_1','-v',f'{root}:/work','-w','/work',
                 '--entrypoint','python','indoorcam:pn41','-m','scripts.probe_yolo_pose','--models',model,
                 '--backend','coral','--model-root','evaluations/yolo-comparison/models',
                 '--images-root','evaluations/pose-comparison/review-frames',
                 '--out',f'evaluations/yolo-comparison/paired/{model}.json']
            try:
                with (out/f'paired-{model}.log').open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['sudo','-n','docker','rm','-f',name],capture_output=True);rc=124
            paired.append({'model':model,'returncode':rc});manifest['paired']=paired;save()
            print('PAIRED',paired[-1],flush=True)
        # Same-session baselines use exactly the previously tested weights.
        shutil.copy2(root/'evaluations/pose-comparison/models/movenet_single_pose_thunder_ptq_edgetpu.tflite',
                     out/'models/movenet_single_pose_thunder_ptq_edgetpu.tflite')
        manifest['full_runner_returncode']=module('run_pose_comparison_pn41','--models','lightning','thunder',*viable,
            '--model-root','evaluations/yolo-comparison','--output','evaluations/yolo-comparison/results',timeout=20000)
    finally:
        if running:subprocess.run(compose+['start','analytics'],check=True)
        manifest['finished']=time.time();save()


if __name__=='__main__':main()
