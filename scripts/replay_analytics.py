#!/usr/bin/env python3
"""Evaluate rule/config changes on recorded model outputs, without another inference run.

This cannot recover missed detections/keypoints and is not an accuracy benchmark.
Explicit reset timestamps prevent a compilation edit from linking unrelated people.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from indoorcam.analytics import Analytics
from indoorcam.config import Config


def replay(states, cfg, cuts=()):
    analytics = Analytics(cfg)
    pending = iter(sorted(cuts))
    cut = next(pending, float("inf"))
    results = []
    for state in states:
        now = state["time"]
        while now >= cut:
            analytics.reset()
            cut = next(pending, float("inf"))
        detections = [{"bbox": p["bbox"], "score": p["score"], "keypoints": np.array(p["keypoints"])}
                      for p in state["people"]]
        result = analytics.update(detections, now, (cfg.height, cfg.width, 3))
        result["camera_id"] = cfg.camera_id
        result["truncated"] = state.get("truncated", False)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", default="config/example.yaml")
    parser.add_argument("--reset-at", type=float, nargs="*", default=[])
    args = parser.parse_args()
    cfg = Config.load(args.config)
    states = [json.loads(line) for line in args.input.read_text().splitlines()]
    result = replay(states, cfg, args.reset_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for state in result:
            handle.write(json.dumps(state)+"\n")
    print(json.dumps({"frames": len(result), "reset_at": args.reset_at,
                      "events": [event for state in result for event in state["events"]]}, indent=2))


if __name__ == "__main__":
    main()
