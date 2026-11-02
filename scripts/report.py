#!/usr/bin/env python3
"""Summarize a replay; create a contact sheet for human visual review."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="artifacts/detections.jsonl")
    parser.add_argument("--video", default="artifacts/annotated.mp4")
    parser.add_argument("--sheet", default="artifacts/contact-sheet.jpg")
    args = parser.parse_args()
    states = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines()]
    counts = Counter(s["count"] for s in states)
    postures = Counter()
    for state in states:
        postures.update(state["postures"])
    result = {"frames": len(states), "count_histogram": dict(counts), "posture_observations": dict(postures),
              "fall_events": [e for state in states for e in state["events"]],
              "note": "Observations, not ground-truth accuracy. Human review required."}
    print(json.dumps(result, indent=2))
    cap = cv2.VideoCapture(args.video)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for t in np.linspace(1, max(1, duration-1), 8):
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ok, frame = cap.read()
        if not ok:
            continue
        tile = cv2.resize(frame, (640,360))
        cv2.putText(tile, f"t={t:.1f}s", (10,350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        tiles.append(tile)
    cap.release()
    if len(tiles) == 8:
        cv2.imwrite(args.sheet, np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)]))


if __name__ == "__main__":
    main()
