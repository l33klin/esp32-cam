"""Offline adapter for normalized, raw 17-keypoint Ultralytics Pose exports."""
import time

import cv2
import numpy as np

from indoorcam.inference import make_interpreter


def letterbox(frame, size):
    h, w = frame.shape[:2]
    ratio = min(size/w, size/h)
    rw, rh = round(w*ratio), round(h*ratio)
    left, top = (size-rw)//2, (size-rh)//2
    rgb = np.full((size, size, 3), 114, np.uint8)
    rgb[top:top+rh, left:left+rw] = cv2.cvtColor(cv2.resize(frame, (rw, rh)), cv2.COLOR_BGR2RGB)
    return rgb, (left, top, w/rw, h/rh)


def decode(raw, size, transform, shape, min_height=.06, nms_iou=.7):
    """Return NMS candidates; output xywh/kpt coordinates are normalized."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 56:
        raise ValueError(f'Expected raw [1,56,N] single-class COCO Pose output, got {raw.shape}')
    left, top, sx, sy = transform
    h, w = shape[:2]
    candidates = []
    for row in raw[0].T[raw[0, 4] >= .05]:
        if not np.all(np.isfinite(row)):
            continue
        cx, cy, bw, bh = row[:4]*size
        box = np.array([(cx-bw/2-left)*sx, (cy-bh/2-top)*sy,
                        (cx+bw/2-left)*sx, (cy+bh/2-top)*sy]).clip([0, 0, 0, 0], [w-1, h-1, w-1, h-1])
        if box[2]-box[0] < 10 or box[3]-box[1] < max(10, h*min_height):
            continue
        points = row[5:].reshape(17, 3).copy()
        points[:, 0] = (points[:, 0]*size-left)*sx
        points[:, 1] = (points[:, 1]*size-top)*sy
        outside = (points[:, 0]<0)|(points[:, 0]>=w)|(points[:, 1]<0)|(points[:, 1]>=h)
        points[outside, 2] = 0
        candidates.append({'bbox': box.tolist(), 'score': float(row[4]), 'keypoints': points})
    if not candidates:
        return []
    xywh = [[p['bbox'][0], p['bbox'][1], p['bbox'][2]-p['bbox'][0],
             p['bbox'][3]-p['bbox'][1]] for p in candidates]
    indices = cv2.dnn.NMSBoxes(xywh, [p['score'] for p in candidates], .05, nms_iou)
    return [candidates[int(i)] for i in np.asarray(indices).reshape(-1)][:50]


class YoloPose:
    def __init__(self, cfg, path, backend='coral'):
        self.cfg, self.backend = cfg, backend
        if backend=='coral':
            self.interpreter = make_interpreter(path, 'pci:0')
        else:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            self.interpreter = Interpreter(model_path=str(path),num_threads=2)
            self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        _, h, w, channels = self.input['shape']
        if h != w or channels != 3:
            raise ValueError('Expected square NHWC RGB model')
        self.size = int(w)
        self.outputs = self.interpreter.get_output_details()
        if len(self.outputs) != 1:
            raise ValueError('Expected one raw head output')

    def predict(self, frame):
        image, transform = letterbox(frame, self.size)
        real = image.astype(np.float32)/255
        if np.issubdtype(self.input['dtype'], np.integer):
            scale, zero = self.input['quantization']
            if scale <= 0:
                raise ValueError('Invalid input quantization')
            limits = np.iinfo(self.input['dtype'])
            real = np.round(real/scale+zero).clip(limits.min, limits.max)
        self.interpreter.set_tensor(self.input['index'], real[None].astype(self.input['dtype']))
        tick = time.perf_counter()
        self.interpreter.invoke()
        invoke_ms = (time.perf_counter()-tick)*1000
        detail = self.outputs[0]
        raw = self.interpreter.get_tensor(detail['index'])
        if np.issubdtype(detail['dtype'], np.integer):
            scale, zero = detail['quantization']
            if scale <= 0:
                raise ValueError('Invalid output quantization')
            raw = (raw.astype(np.float32)-zero)*scale
        candidates = decode(raw, self.size, transform, frame.shape, self.cfg.min_person_height)
        people = [p for p in candidates if p['score'] >= self.cfg.threshold and
                  np.count_nonzero(p['keypoints'][:, 2]>=self.cfg.keypoint_threshold) >= 3][:self.cfg.max_people]
        return people, {'detector_invoke_ms': 0., 'pose_invoke_ms': invoke_ms,
            'joint_network_invoke_ms': invoke_ms,
            'candidates': [{**p, 'keypoints': p['keypoints'].tolist()} for p in candidates]}
