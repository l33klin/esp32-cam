import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit
import cv2
import numpy as np
from .analytics import Analytics
from .config import Config
from .inference import Inference
from .mqtt import HomeAssistant
from .video import LatestCapture, FrameHub, RtspWriter, annotate, fit, start_api

LOG = logging.getLogger(__name__)


def offline_frames(cfg, stop):
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        cap.release()
        raise ValueError("Input file has no usable FPS")
    if cfg.fps > source_fps:
        cap.release()
        raise ValueError("Offline output fps cannot exceed input fps")
    index, next_time = 0, 0.0
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index/source_fps
            index += 1
            if timestamp+1e-6 < next_time:
                continue
            yield next_time, frame, 0
            next_time += 1/cfg.fps
    finally:
        cap.release()


def live_frames(cfg, stop, hub, mqtt):
    reader = LatestCapture(cfg.source, cfg.loop, rtsp_size=(cfg.width, cfg.height))
    serial, generation = -1, -1
    try:
        while not stop.is_set():
            start = time.monotonic()
            item = reader.get()
            if item and item[0] != serial and start-item[1] < 5:
                serial, ts, frame, gen = item
                if gen != generation:
                    generation = gen
                yield ts, frame, generation
            else:
                if not item or start-item[1] >= 5:
                    hub.invalidate()
                    if mqtt:
                        mqtt.offline()
                stop.wait(0.05)
            stop.wait(max(0, 1/cfg.fps-(time.monotonic()-start)))
    finally:
        reader.close()


def run(cfg, max_frames=0):
    cfg.validate()
    inference, analytics, hub = Inference(cfg), Analytics(cfg), FrameHub()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    mqtt = api = rtsp = writer = records = None
    count, durations, events, previous_generation = 0, [], [], None
    started = time.monotonic()
    try:
        if cfg.mqtt_host:
            mqtt = HomeAssistant(cfg)
        if cfg.api_port:
            api = start_api(hub, cfg.api_host, cfg.api_port)
        if cfg.rtsp_output:
            rtsp = RtspWriter(cfg, hub)
        if cfg.output_file:
            Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(cfg.output_file, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (cfg.width, cfg.height))
            if not writer.isOpened():
                raise RuntimeError("Cannot open output video")
        if cfg.jsonl:
            Path(cfg.jsonl).parent.mkdir(parents=True, exist_ok=True)
            records = open(cfg.jsonl, "w")
        live = cfg.realtime or bool(urlsplit(cfg.source).scheme)
        if cfg.loop and not live:
            raise ValueError("loop requires realtime mode")
        if (cfg.rtsp_output or cfg.mqtt_host) and not live:
            raise ValueError("RTSP/MQTT require realtime mode to preserve event timing")
        frames = live_frames(cfg, stop, hub, mqtt) if live else offline_frames(cfg, stop)
        try:
            for timestamp, frame, generation in frames:
                tick = time.monotonic()
                if generation != previous_generation:
                    analytics.reset()
                    previous_generation = generation
                frame = fit(frame, cfg.width, cfg.height)
                detections = inference.detect(frame)
                truncated = len(detections) >= cfg.max_people
                for person in detections:
                    person["keypoints"] = inference.keypoints(frame, person["bbox"])
                detections = [person for person in detections if inference.verified(person)]
                state = analytics.update(detections, timestamp, frame.shape)
                state["truncated"] = truncated
                state["camera_id"] = cfg.camera_id
                frame = annotate(frame, state, cfg)
                hub.set(frame, state)
                if mqtt:
                    mqtt.update(state)
                if writer:
                    writer.write(frame)
                if records:
                    records.write(json.dumps(state)+"\n")
                    records.flush()
                durations.append(time.monotonic()-tick)
                events.extend(state["events"])
                count += 1
                if count % 80 == 0:
                    LOG.info("frames=%d people=%d inference_ms=%.1f", count, state["count"], 1000*durations[-1])
                if max_frames and count >= max_frames:
                    break
        finally:
            frames.close()
    finally:
        stop.set()
        hub.invalidate()
        if rtsp:
            rtsp.close()
        if mqtt:
            mqtt.close()
        if api:
            api.shutdown()
            api.server_close()
        if writer:
            writer.release()
        if records:
            records.close()
    summary = {"frames": count, "backend": cfg.backend, "elapsed_s": round(time.monotonic()-started, 2),
               "detector_model": cfg.detector_model, "detector_tiles": cfg.detector_tiles,
               "pose_backend": cfg.backend if cfg.pose_backend == "auto" else cfg.pose_backend,
               "pose_model": cfg.pose_model,
               "processing_ms_mean": round(float(np.mean(durations))*1000, 2) if durations else None,
               "processing_ms_p95": round(float(np.percentile(durations, 95))*1000, 2) if durations else None,
               "fall_events": len(events), "events": events}
    if cfg.jsonl:
        Path(cfg.jsonl+".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Indoor camera analytics")
    parser.add_argument("--config", default="config/example.yaml")
    for arg in ("source", "backend", "output-file", "jsonl"):
        parser.add_argument("--"+arg)
    parser.add_argument("--detector-tiles", type=int)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    for key in ("source", "backend", "output_file", "jsonl", "detector_tiles"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    cfg.realtime = cfg.realtime or args.realtime
    cfg.loop = cfg.loop or args.loop
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(cfg, args.max_frames)


if __name__ == "__main__":
    main()
