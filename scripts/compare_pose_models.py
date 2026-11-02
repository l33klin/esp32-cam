"""Offline Coral pose comparison. Never publishes MQTT, HTTP or RTSP.

Run in pn41's compatible TFLite 2.5 container, with exclusive Coral access.
PoseNet decoding follows google-coral/project-posenet (Apache-2.0).
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

from indoorcam.analytics import Analytics
from indoorcam.config import Config
from indoorcam.inference import Inference, make_interpreter
from indoorcam.video import annotate, fit


MODELS = {
    "lightning": ("crop", "pose_edgetpu.tflite"),
    "thunder": ("crop", "movenet_single_pose_thunder_ptq_edgetpu.tflite"),
    "posenet353": ("mobilenet", "posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite"),
    "posenet481": ("mobilenet", "posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"),
    "posenet721": ("mobilenet", "posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite"),
    "resnet416": ("resnet", "posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite"),
    "resnet640": ("resnet", "posenet_resnet_50_640_480_16_quant_edgetpu_decoder.tflite"),
    "efficientdet0_thunder": ("crop", "movenet_single_pose_thunder_ptq_edgetpu.tflite"),
    "efficientdet1_thunder": ("crop", "movenet_single_pose_thunder_ptq_edgetpu.tflite"),
    "efficientdet2_thunder": ("crop", "movenet_single_pose_thunder_ptq_edgetpu.tflite"),
    "posenet721_thunder": ("hybrid", "posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite"),
    "efficientdet1_full_thunder": ("crop", "movenet_single_pose_thunder_ptq_edgetpu.tflite"),
    "efficientdet1_letterbox_thunder": ("crop", "movenet_single_pose_thunder_ptq_edgetpu.tflite"),
}
DETECTORS = {f"efficientdet{i}_thunder":f"efficientdet_lite{i}_{size}_ptq_edgetpu.tflite"
             for i,size in ((0,320),(1,384),(2,448))}
DETECTORS['efficientdet1_full_thunder']='efficientdet_lite1_384_ptq_edgetpu.tflite'
DETECTORS['efficientdet1_letterbox_thunder']='efficientdet_lite1_384_ptq_edgetpu.tflite'
MODELS.update({f'{model}_{size}':('yolo',f'{model}_{size}_int8_edgetpu.tflite')
               for model in ('yolov8n','yolo11n','yolo26n') for size in (320,480)})
MODELS['yolov8n_480_thunder']=('hybrid_yolo','yolov8n_480_int8_edgetpu.tflite')
MODELS['yolov8n_480_cal010']=('yolo','yolov8n_480_int8_edgetpu.tflite')
MODELS.update({f'{m}_512':('yolo',f'{m}_512_int8_edgetpu.tflite') for m in ('yolo11n','yolo26n')})


class TimedInterpreter:
    def __init__(self, interpreter):
        self.interpreter, self.ms = interpreter, 0.0

    def invoke(self):
        tick = time.perf_counter()
        self.interpreter.invoke()
        self.ms += (time.perf_counter()-tick)*1000

    def __getattr__(self, key):
        return getattr(self.interpreter, key)


class CropMoveNet(Inference):
    def __init__(self, cfg, pose_path, detector_path=None):
        self.cfg = cfg
        self.detector_path = detector_path or Path(cfg.model_dir)/"mobiledet_edgetpu.tflite"
        self.detector = TimedInterpreter(make_interpreter(self.detector_path, "pci:0"))
        self.pose = TimedInterpreter(make_interpreter(pose_path, "pci:1"))

    def predict(self, frame):
        self.detector.ms = self.pose.ms = 0.0
        candidates = self.detect(frame)
        for person in candidates:
            person["keypoints"] = self.keypoints(frame, person["bbox"])
        people = [p for p in candidates if self.verified(p)]
        return people, {"detector_invoke_ms": self.detector.ms, "pose_invoke_ms": self.pose.ms,
                        "candidates": serialize(candidates)}


def letterbox_rgb(frame, width, height, centered=True):
    fh, fw = frame.shape[:2]
    scale = min(width/fw, height/fh)
    rw, rh = round(fw*scale), round(fh*scale)
    left, top = ((width-rw)//2, (height-rh)//2) if centered else (0, 0)
    image = np.zeros((height, width, 3), np.uint8)
    image[top:top+rh, left:left+rw] = cv2.cvtColor(cv2.resize(frame, (rw, rh)), cv2.COLOR_BGR2RGB)
    return image, (left, top, fw/rw, fh/rh)


class LetterboxEfficientDetThunder(CropMoveNet):
    """Full-frame, aspect-preserving EfficientDet adapter.

    Threshold .20 is the highest tied minimum of count error on the predefined
    calibration subset of the detector-only diagnostic, before pose verification.
    """
    def detect(self, frame):
        inp = self.detector.get_input_details()[0]
        _, ih, iw, _ = inp['shape']
        image, (_, _, sx, sy) = letterbox_rgb(frame, int(iw), int(ih), centered=False)
        self.detector.set_tensor(inp['index'], image[None].astype(inp['dtype']))
        self.detector.invoke()
        boxes, classes, scores, count = [self.detector.get_tensor(d['index'])
                                          for d in self.detector.get_output_details()]
        h, w = frame.shape[:2]
        people = []
        for i in range(min(int(count.flat[0]), boxes.shape[1])):
            score = float(scores[0, i])
            if int(classes[0, i]) != 0 or score < self.cfg.threshold:
                continue
            y1, x1, y2, x2 = boxes[0, i]
            box = np.array([x1*iw*sx, y1*ih*sy, x2*iw*sx, y2*ih*sy]).clip(
                [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
            if box[2]-box[0] < 10 or box[3]-box[1] < max(10, h*self.cfg.min_person_height):
                continue
            people.append({'bbox': box.tolist(), 'score': score})
        if not people:
            return []
        xywh = [[p['bbox'][0], p['bbox'][1], p['bbox'][2]-p['bbox'][0],
                 p['bbox'][3]-p['bbox'][1]] for p in people]
        indices = cv2.dnn.NMSBoxes(xywh, [p['score'] for p in people], self.cfg.threshold, .4)
        return [people[int(i)] for i in np.asarray(indices).reshape(-1)][:self.cfg.max_people]


def unletterbox(points_yx, confidence, transform, shape):
    left, top, sx, sy = transform
    points = np.stack([(points_yx[:, 1]-left)*sx, (points_yx[:, 0]-top)*sy, confidence], axis=1)
    h, w = shape[:2]
    outside = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
    points[outside, 2] = 0
    return points


def skeleton_box(points, shape, confidence=0.2):
    visible = points[points[:, 2] >= confidence, :2]
    if len(visible) < 3:
        return None
    lo, hi = visible.min(axis=0), visible.max(axis=0)
    margin = np.maximum((hi-lo)*0.1, 3)
    h, w = shape[:2]
    return np.concatenate([lo-margin, hi+margin]).clip([0, 0, 0, 0], [w-1, h-1, w-1, h-1]).tolist()


class MultiPose:
    def __init__(self, cfg, path, decoder, threshold):
        from tflite_runtime.interpreter import Interpreter, load_delegate
        self.cfg, self.threshold = cfg, threshold
        self.interpreter = TimedInterpreter(Interpreter(model_path=str(path), num_threads=2,
            experimental_delegates=[load_delegate("libedgetpu.so.1", {"device": "pci:0"}),
                                    load_delegate(str(decoder))]))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        _, self.height, self.width, _ = self.input["shape"]
        self.outputs = self.interpreter.get_output_details()

    def predict(self, frame):
        image, transform = letterbox_rgb(frame, int(self.width), int(self.height))
        if self.input["dtype"] == np.float32:
            image = image.astype(np.float32)/128.0-1.0
        self.interpreter.set_tensor(self.input["index"], image[None].astype(self.input["dtype"]))
        self.interpreter.ms = 0
        self.interpreter.invoke()
        coords, confidences, scores, num = [self.interpreter.get_tensor(d["index"])[0] for d in self.outputs]
        candidates, people = [], []
        for i in range(min(int(np.asarray(num).flat[0]), len(scores))):
            points = unletterbox(coords[i], confidences[i], transform, frame.shape)
            box = skeleton_box(points, frame.shape)
            if box is None:
                continue
            p = {"bbox": box, "score": float(scores[i]), "keypoints": points}
            candidates.append(p)
            if p["score"] < self.threshold or box[3]-box[1] < frame.shape[0]*self.cfg.min_person_height:
                continue
            if np.count_nonzero(points[:, 2] >= self.cfg.keypoint_threshold) < 3:
                continue
            people.append(p)
        people.sort(key=lambda p: p["score"], reverse=True)
        return people[:self.cfg.max_people], {"detector_invoke_ms": 0.0, "pose_invoke_ms": self.interpreter.ms,
            "candidates": serialize(candidates)}


class ProposalThunder:
    """Experimental composition: PoseNet whole-frame proposals, Thunder crops.

    Proposal threshold .20 is the highest tied minimum count error on the
    predefined calibration subset; no review-frame tuning is used.
    """
    def __init__(self,cfg,proposal_path,decoder,refiner_path):
        self.cfg=cfg
        self.proposals=MultiPose(cfg,proposal_path,decoder,.20)
        self.interpreter=self.proposals.interpreter
        self.refiner=Inference.__new__(Inference)
        self.refiner.cfg=cfg
        self.refiner.pose=TimedInterpreter(make_interpreter(refiner_path,'pci:1'))
        self.refiner_path=refiner_path

    def predict(self,frame):
        candidates,_=self.proposals.predict(frame)
        self.refiner.pose.ms=0
        for p in candidates:
            p['keypoints']=self.refiner.keypoints(frame,p['bbox'])
        people=[p for p in candidates if self.refiner.verified(p)]
        return people,{'detector_invoke_ms':self.proposals.interpreter.ms,
                       'pose_invoke_ms':self.refiner.pose.ms,'candidates':serialize(candidates)}


class YoloProposalThunder:
    """YOLO480 proposals, then Thunder on the second Coral.

    Proposal threshold .10 was selected by count error on the predefined
    calibration subset of the base INT8 YOLO JPEG probe, before refinement.
    """
    def __init__(self,cfg,path,refiner_path):
        from scripts.yolo_pose_adapter import YoloPose
        self.proposals=YoloPose(cfg,path,'coral')
        self.proposals.threshold=cfg.threshold
        self.interpreter=self.proposals.interpreter
        self.refiner=Inference.__new__(Inference);self.refiner.cfg=cfg
        self.refiner.pose=TimedInterpreter(make_interpreter(refiner_path,'pci:1'))
        self.refiner_path=refiner_path;self.cfg=cfg

    def predict(self,frame):
        _,extra=self.proposals.predict(frame)
        candidates=[{'bbox':p['bbox'],'score':p['score']} for p in extra['candidates']
                    if p['score']>=self.cfg.threshold][:self.cfg.max_people]
        self.refiner.pose.ms=0
        for p in candidates:p['keypoints']=self.refiner.keypoints(frame,p['bbox'])
        people=[p for p in candidates if self.refiner.verified(p)]
        return people,{'detector_invoke_ms':extra['joint_network_invoke_ms'],
                       'pose_invoke_ms':self.refiner.pose.ms,'candidates':serialize(candidates)}


def serialize(people):
    return [{**p, "keypoints": p["keypoints"].tolist()} for p in people]


def model_path(name, model_root, production_root):
    family, filename = MODELS[name]
    if name == "lightning":
        return production_root/filename
    if family in ("crop", "yolo", "hybrid_yolo"):
        return model_root/"models"/filename
    return model_root/"project-posenet/models"/('mobilenet' if family=='hybrid' else family)/filename


def stats(values):
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95))} if len(a) else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("evaluations/pose-comparison"))
    parser.add_argument("--production-models", type=Path, default=Path("models"))
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--pose-threshold", type=float, default=0.25)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--cuts", type=float, nargs="*", default=[])
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--input-max-width", type=int, default=0, help="Optional source detail limit before fitting output, for VGA-width checks")
    parser.add_argument('--yolo-backend', choices=['coral','cpu'], default='coral')
    parser.add_argument('--yolo-model-file', type=Path, help='Explicit compiled variant for compatibility probes')
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must be a new directory")
    args.out.mkdir(parents=True)
    cv2.setNumThreads(2)
    cfg = Config(source=str(args.source), model_dir=str(args.production_models), fps=args.fps,
                 backend='cpu' if MODELS[args.model][0]=='yolo' and args.yolo_backend=='cpu' else 'coral',
                 coral_device="pci:0", pose_device="pci:1", camera_id="offline_pose_comparison",
                 detector_tiles=1 if ('_full_' in args.model or '_letterbox_' in args.model) else 2,
                 threshold=(.10 if MODELS[args.model][0]=='hybrid_yolo' or args.model.endswith('_cal010') else
                    (args.pose_threshold if MODELS[args.model][0]=='yolo' else (.20 if '_letterbox_' in args.model else .45))))
    cfg.validate()
    path = model_path(args.model, args.model_root, args.production_models)
    detector_path = args.model_root/"models"/DETECTORS[args.model] if args.model in DETECTORS else None
    decoder=args.model_root/"project-posenet/posenet_lib/x86_64/posenet_decoder.so"
    if MODELS[args.model][0]=='hybrid_yolo':
        model=YoloProposalThunder(cfg,path,args.model_root/'models/movenet_single_pose_thunder_ptq_edgetpu.tflite')
    elif MODELS[args.model][0]=='yolo':
        from scripts.yolo_pose_adapter import YoloPose
        if args.yolo_model_file:
            path=args.yolo_model_file
        elif args.yolo_backend=='cpu':
            path=path.with_name(path.name.replace('_int8_edgetpu','_fp32'))
        model=YoloPose(cfg,path,args.yolo_backend)
    elif MODELS[args.model][0]=='hybrid':
        model=ProposalThunder(cfg,path,decoder,args.model_root/'models/movenet_single_pose_thunder_ptq_edgetpu.tflite')
    elif MODELS[args.model][0]=='crop':
        adapter=LetterboxEfficientDetThunder if '_letterbox_' in args.model else CropMoveNet
        model=adapter(cfg,path,detector_path)
    else:
        model=MultiPose(cfg,path,decoder,args.pose_threshold)
    interpreter = model.pose if isinstance(model, CropMoveNet) else model.interpreter
    model_details = {"input_shape":interpreter.get_input_details()[0]["shape"].tolist(),
                     "input_dtype":str(interpreter.get_input_details()[0]["dtype"]),
                     "file_bytes":path.stat().st_size}
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError("Input video failed to open")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    if not ok or source_fps <= 0:
        raise RuntimeError("Input has no decodable frame/fps")
    frame = fit(frame, cfg.width, cfg.height)
    for _ in range(3):
        model.predict(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = encode_log = None
    if not args.no_video:
        encode_log = (args.out/"encode.log").open("w")
        writer = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(args.out/"annotated.mp4")],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_log)
    analytics, cuts = Analytics(cfg), iter(sorted(args.cuts))
    next_cut = next(cuts, float("inf"))
    index, sampled = 0, 0
    timings, detector_times, pose_times, analytics_times = [], [], [], []
    counts, postures, raw_postures, events = Counter(), Counter(), Counter(), []
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    record = (args.out/"detections.jsonl").open("w")
    try:
        while True:
            ok, original = cap.read()
            if not ok:
                break
            media_time = index/source_fps
            index += 1
            if args.max_seconds and media_time >= args.max_seconds:
                break
            if media_time+1e-6 < sampled/args.fps:
                continue
            timestamp = sampled/args.fps
            while timestamp >= next_cut:
                analytics.reset()
                next_cut = next(cuts, float("inf"))
            if args.input_max_width and original.shape[1] > args.input_max_width:
                h,w = original.shape[:2]
                original = cv2.resize(original,(args.input_max_width,round(h*args.input_max_width/w)))
            image = fit(original, cfg.width, cfg.height)
            tick = time.perf_counter()
            people, extra = model.predict(image)
            infer_ms = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            state = analytics.update(people, timestamp, image.shape)
            analytics_ms = (time.perf_counter()-tick)*1000
            state.update(model=args.model, camera_id=cfg.camera_id, source_time=media_time,
                         inference_ms=infer_ms, analytics_ms=analytics_ms, truncated=len(people)>=cfg.max_people,
                         **extra)
            record.write(json.dumps(state)+"\n")
            if writer:
                drawn = annotate(image, state, cfg)
                cv2.putText(drawn, f"{args.model}   t={timestamp:.3f}s", (8, cfg.height-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.stdin.write(drawn.tobytes())
            timings.append(infer_ms)
            detector_times.append(extra["detector_invoke_ms"])
            pose_times.append(extra["pose_invoke_ms"])
            analytics_times.append(analytics_ms)
            counts[state["count"]] += 1
            postures.update(state["postures"])
            raw_postures.update(p["raw_posture"] for p in state["people"])
            events.extend(state["events"])
            sampled += 1
            if sampled % 160 == 0:
                print(json.dumps({"model":args.model,"frames":sampled,"media_seconds":timestamp,"inference_ms":round(infer_ms,2)}),flush=True)
            if args.max_frames and sampled >= args.max_frames:
                break
    finally:
        record.close()
        cap.release()
        if writer:
            writer.stdin.close()
            if writer.wait(timeout=60):
                raise RuntimeError("Video encoding failed; see encode.log")
            encode_log.close()
    elapsed, cpu = time.perf_counter()-wall_start, time.process_time()-cpu_start
    summary = {"model":args.model,"input":str(args.source),"frames":sampled,"sample_fps":args.fps,
        "elapsed_s":elapsed,"pipeline_fps":sampled/elapsed,"parent_cpu_seconds":cpu,"parent_cpu_percent":cpu/elapsed*100,
        "peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        "inference_ms":stats(timings),"detector_invoke_ms":stats(detector_times),"pose_invoke_ms":stats(pose_times),
        "analytics_ms":stats(analytics_times),"posture_observations":dict(postures),"raw_posture_observations":dict(raw_postures),
        "count_distribution":dict(counts),"fall_candidates":events,"scene_cuts":args.cuts,
        "pose_threshold":args.pose_threshold,"input_max_width":args.input_max_width,"coral_devices":(0 if MODELS[args.model][0]=='yolo' and args.yolo_backend=='cpu' else (2 if MODELS[args.model][0] in ('crop','hybrid','hybrid_yolo') else 1)),
        "model_file":str(path),"model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"model_details":model_details,"config":asdict(cfg),
        "timing_notes":"Three warmups excluded. invoke includes delegated and CPU operators; pipeline includes decode, resize, analytics and H264 writing; CPU percent is parent process only."}
    if isinstance(model,CropMoveNet):
        summary.update(detector_file=str(model.detector_path),detector_sha256=hashlib.sha256(model.detector_path.read_bytes()).hexdigest())
        summary['detector_preprocessing']='top-left letterbox RGB' if isinstance(model,LetterboxEfficientDetThunder) else 'stretch RGB per crop'
    if isinstance(model,(ProposalThunder,YoloProposalThunder)):
        summary.update(proposal_threshold=model.proposals.threshold,refiner_file=str(model.refiner_path),
                       refiner_sha256=hashlib.sha256(model.refiner_path.read_bytes()).hexdigest())
    summary['config_note']='config contains common baseline/rule defaults; the actual evaluated networks are identified by model/model_file/detector_file, not config.detector_model.'
    if MODELS[args.model][0]=='yolo':
        summary.update(inference_backend=args.yolo_backend,head='normalized raw one-to-many COCO17',
                       nms_iou=.7,confidence_threshold=cfg.threshold,minimum_visible_keypoints=3,
                       invoke_note='joint person detection + pose network; CPU ops may remain')
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','fall_candidates')},indent=2),flush=True)


if __name__ == "__main__":
    main()
