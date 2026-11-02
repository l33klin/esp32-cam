"""Run sequential isolated Coral evaluations, restoring analytics in finally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluations/pose-comparison/results")
    parser.add_argument("--models", nargs="+", default=["posenet353","posenet481","posenet721","lightning","thunder","resnet416","resnet640"])
    parser.add_argument("--restore-analytics", action="store_true")
    parser.add_argument("--input-max-width",type=int,default=0)
    parser.add_argument('--model-root', default='evaluations/pose-comparison')
    parser.add_argument("--wait-manifest",type=Path,help="Wait for another batch's finished marker, without sharing TPUs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.wait_manifest:
        deadline=time.monotonic()+3600
        while True:
            try:
                finished=json.loads((root/args.wait_manifest).read_text()).get('finished')
            except (OSError,ValueError):
                finished=False
            if finished:break
            if time.monotonic()>deadline:raise TimeoutError('Previous batch did not finish within one hour')
            time.sleep(5)
    output = root/args.output
    output.mkdir(parents=True,exist_ok=True)
    compose = ["sudo","-n","docker","compose","--env-file",str(root/"deploy/.env"),
               "-f",str(root/"deploy/compose.yaml"),"-f",str(root/"deploy/compose.pn41.yaml")]
    running = subprocess.check_output(["sudo","-n","docker","inspect","--format","{{.State.Running}}","indoorcam-analytics-1"],text=True).strip()=="true"
    restore = running or args.restore_analytics
    videos = {"ring":root/"test.mkv","falls":root/"evaluations/pose-comparison/falls-full.mp4"}
    cuts = json.loads((root/"evaluations/pose-comparison/scene-cuts.json").read_text())["cuts"]
    manifest = {"started":time.time(),"models":args.models,"videos":{k:str(v) for k,v in videos.items()},
                "restore_analytics":restore,"sample_fps":8,"pose_threshold":0.25,"input_max_width":args.input_max_width,"fall_video_scene_cuts":cuts,"runs":[]}
    def save():
        (output/"run-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save()
    subprocess.run(compose+["stop","analytics"],check=True)
    try:
        for model in args.models:
            for video,source in videos.items():
                run_dir = output/model/video
                run_dir.parent.mkdir(parents=True,exist_ok=True)
                if (run_dir/"summary.json").exists():
                    print("SKIP completed",model,video,flush=True)
                    continue
                name = f"indoorcam-pose-eval-{model}-{video}"
                cmd = ["sudo","-n","docker","run","--rm","--name",name,"--user","0","--network","none",
                    "--device","/dev/apex_0","--device","/dev/apex_1","-v",f"{root}:/work","-w","/work",
                    "--entrypoint","python","indoorcam:pn41","-m","scripts.compare_pose_models",
                    "--model",model,"--model-root",args.model_root,"--source",str(Path("/work")/source.relative_to(root)),
                    "--out",str(Path("/work")/run_dir.relative_to(root))]
                if args.input_max_width:
                    cmd += ["--input-max-width",str(args.input_max_width)]
                if video == "falls":
                    cmd += ["--cuts"]+[str(t) for t in cuts]
                record = {"model":model,"video":video,"start":time.time(),"host_load":os.getloadavg()}
                print("START",model,video,flush=True)
                try:
                    with (run_dir.parent/f"{video}.log").open("w") as log:
                        p = subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                    record["returncode"] = p.returncode
                except subprocess.TimeoutExpired:
                    record["error"] = "timeout_1800s"
                finally:
                    subprocess.run(["sudo","-n","docker","rm","-f",name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                record["end"] = time.time()
                manifest["runs"].append(record)
                save()
                print("END",json.dumps(record),flush=True)
    finally:
        if restore:
            subprocess.run(compose+["start","analytics"],check=True)
        manifest["finished"] = time.time()
        save()


if __name__ == "__main__":
    main()
