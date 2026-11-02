#!/usr/bin/env python3
"""Borrow the project's exclusive Coral devices, restoring its service afterward.

Run on pn41 from this project. Evaluation never publishes MQTT/RTSP.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overrides", type=Path, help="Optional JSON parameters, applied before mandatory offline settings")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.input.resolve()
    output = args.output_dir.resolve()
    relative = output.relative_to(root)
    if not source.is_file() or output.exists():
        parser.error("Input must exist; output_dir must be a new directory under the project")
    output.mkdir(parents=True)
    cfg = json.loads((root/"deploy/runtime/config.json").read_text())
    if args.overrides:
        cfg.update(json.loads(args.overrides.read_text()))
    cfg.update(source="/media/input.mp4", backend="coral", model_dir="/app/models", fps=8,
               camera_id="fall_evaluation", rtsp_output="", mqtt_host="", api_port=0,
               loop=False, realtime=False, output_file=f"/work/{relative}/annotated.mp4",
               jsonl=f"/work/{relative}/detections.jsonl")
    config = output/"config.json"
    config.write_text(json.dumps(cfg, indent=2))
    compose = ["sudo", "-n", "docker", "compose", "--env-file", str(root/"deploy/.env"),
               "-f", str(root/"deploy/compose.yaml"), "-f", str(root/"deploy/compose.pn41.yaml")]
    container = subprocess.check_output(compose+["ps", "--all", "--quiet", "analytics"], text=True).strip()
    running = bool(container) and subprocess.check_output(
        ["sudo", "-n", "docker", "inspect", "--format", "{{.State.Running}}", container], text=True).strip() == "true"
    if running:
        subprocess.run(compose+["stop", "analytics"], check=True)
    try:
        subprocess.run(["sudo", "-n", "docker", "run", "--rm", "--user", "0", "--network", "none",
            "--device", "/dev/apex_0", "--device", "/dev/apex_1", "-v", f"{root}:/work", "-w", "/work",
            "-v", f"{source}:/media/input.mp4:ro", "--entrypoint", "python", "indoorcam:pn41",
            "-m", "indoorcam.app", "--config", f"/work/{relative}/config.json"], check=True)
    finally:
        if running:
            subprocess.run(compose+["start", "analytics"], check=True)


if __name__ == "__main__":
    main()
