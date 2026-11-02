"""Short exclusive-Coral compatibility screen; restores original analytics."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',nargs='+',required=True);args=p.parse_args()
    root=Path.cwd();out=root/'evaluations/yolo-comparison';tag=str(int(time.time()));rows=[]
    compose=['sudo','-n','docker','compose','--env-file','deploy/.env','-f','deploy/compose.yaml','-f','deploy/compose.pn41.yaml']
    running=subprocess.check_output(['sudo','-n','docker','inspect','--format','{{.State.Running}}','indoorcam-analytics-1'],text=True).strip()=='true'
    subprocess.run(compose+['stop','analytics'],check=True)
    try:
        for model,video,source in [(m,v,s) for m in args.models for v,s in
                [('ring','test.mkv'),('falls','evaluations/pose-comparison/falls-full.mp4')]]:
            dest=f'evaluations/yolo-comparison/smoke/{tag}/{model}/{video}'
            cmd=['sudo','-n','docker','run','--rm','--name',f'indoorcam-yolo-smoke-{model}',
                '--network','none','--user','0','--device','/dev/apex_0','--device','/dev/apex_1',
                '-v',f'{root}:/work','-w','/work','--entrypoint','python','indoorcam:pn41',
                '-m','scripts.compare_pose_models','--model',model,'--model-root','evaluations/yolo-comparison',
                '--source',source,'--out',dest,'--no-video','--max-frames','16']
            try:
                with (out/f'smoke-{model}-{video}-{tag}.log').open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['sudo','-n','docker','rm','-f',f'indoorcam-yolo-smoke-{model}'],capture_output=True);rc=124
            row={'model':model,'video':video,'returncode':rc,'output':dest}
            summary=root/dest/'summary.json'
            if summary.exists():
                data=json.loads(summary.read_text());row.update(inference_ms=data['inference_ms'],pipeline_fps=data['pipeline_fps'])
            rows.append(row);print(row,flush=True)
    finally:
        if running:subprocess.run(compose+['start','analytics'],check=True)
        (out/f'smoke-{tag}.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
