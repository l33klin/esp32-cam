"""Offline temporal adapters. No imports or changes to production inference.

Research implementations/weights remain under ignored artifacts; no network at
inference time. Only restricted tensor checkpoints are loaded with PyTorch.
"""
from collections import deque
from pathlib import Path
import sys

import numpy as np

CLASSES = ('Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down')
COCO13 = [0] + list(range(5, 17))


def tsstg_inputs(sequence):
    """Match upstream image-normalize -> per-frame xy min/max -> neck node."""
    points = np.asarray(sequence, np.float32)[:, COCO13].copy()
    points[..., :2] /= [960, 540]
    xy = points[..., :2]
    lo, hi = xy.min(axis=1, keepdims=True), xy.max(axis=1, keepdims=True)
    if np.any(hi - lo < 1e-7):
        raise ValueError('Degenerate skeleton extent')
    points[..., :2] = (xy - lo) / (hi - lo) * 2 - 1
    neck = (points[:, 1:2] + points[:, 2:3]) / 2
    points = np.concatenate([points, neck], axis=1).transpose(2, 0, 1)[None]
    return {'points': np.ascontiguousarray(points),
            'motion': np.ascontiguousarray(np.diff(points[:, :2], axis=2))}


def mcfd_features(sequence, velocity=True):
    """Match released pose_to_feature_vector with min_confidence=.05."""
    points = np.asarray(sequence, np.float32)
    xy, confidence = points[..., :2], points[..., 2:3]
    normalized = np.zeros_like(xy)
    for t in range(len(points)):
        valid = confidence[t, :, 0] > .05
        if np.count_nonzero(valid) < 2:
            continue
        lo, hi = xy[t, valid].min(axis=0), xy[t, valid].max(axis=0)
        center = (lo + hi) / 2
        scale = max(float(np.max(np.maximum(hi - lo, 1))), 1)
        normalized[t] = (xy[t] - center) / scale
    parts = [normalized.reshape(len(points), -1), confidence.reshape(len(points), -1)]
    if velocity:
        speed = np.zeros_like(normalized)
        speed[1:] = normalized[1:] - normalized[:-1]
        parts.append(speed.reshape(len(points), -1))
    return {'features': np.concatenate(parts, axis=1)[None].astype(np.float32)}


def load_torch_model(kind, root):
    import torch
    root = Path(root).resolve()
    if kind == 'tsstg':
        sys.path.insert(0, str(root / 'research'))
        from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
        model = TwoStreamSpatialTemporalGraph({'strategy': 'spatial'}, 7)
        path = root / 'models/tsstg-model.pth'
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        config = {'frames': 30, 'classes': CLASSES}
    else:
        sys.path.insert(0, str(root / 'research/tcn'))
        from fall.models.pose import build_pose_model
        name = 'mcfd_pose_tcn_attention' if kind == 'mcfd_tcn' else 'mcfd_pose_lstm'
        path = root / 'research/tcn/outputs' / name / 'best.pt'
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        cfg = ckpt['config']
        model = build_pose_model(cfg['model']['name'], ckpt['input_dim'], cfg['model'])
        model.load_state_dict(ckpt['model'], strict=True)
        config = {'frames': cfg['data']['frames'], 'velocity': cfg['data']['use_velocity'],
                  'classes': ['non_fall', 'fall'], 'checkpoint_epoch': ckpt['epoch']}
    return model.eval(), config, path


class TemporalModel:
    def __init__(self, kind, root, backend='onnx'):
        self.kind, self.backend = kind, backend
        root = Path(root)
        if backend == 'onnx':
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(root / 'models' / (kind + '.onnx')),
                sess_options=options, providers=['CPUExecutionProvider'])
        else:
            import torch
            torch.set_num_threads(2)
            self.model, _, _ = load_torch_model(kind, root)

    def inputs(self, sequence):
        return tsstg_inputs(sequence) if self.kind == 'tsstg' else mcfd_features(sequence, self.kind == 'mcfd_tcn')

    def predict(self, sequence):
        feeds = self.inputs(sequence)
        if self.backend == 'onnx':
            output = self.session.run(None, feeds)[0].reshape(-1)
        else:
            import torch
            inputs = [torch.from_numpy(x) for x in feeds.values()]
            with torch.inference_mode():
                output = self.model(tuple(inputs) if self.kind == 'tsstg' else inputs[0]).numpy().reshape(-1)
        if self.kind == 'tsstg':
            index = int(np.argmax(output))
            return {'action': CLASSES[index], 'confidence': float(output[index]),
                    'fall_score': float(output[6]), 'scores': output.tolist(),
                    'fall_top1': index == 6}
        score = float(1 / (1 + np.exp(-np.clip(output[0], -80, 80))))
        return {'action': 'fall' if score >= .5 else 'non_fall', 'confidence': max(score, 1-score),
                'fall_score': score, 'scores': [1-score, score], 'fall_top1': score >= .5}


class History:
    """Uses only observations at or before now; no bridging long gaps/cuts."""
    def __init__(self, seconds, fps, samples):
        self.seconds, self.fps, self.samples = seconds, fps, samples
        self.data = deque()
        self.max_gap = max(.5, 1.5/fps)

    def append(self, timestamp, points):
        points = np.asarray(points, np.float32)
        if points.shape != (17, 3) or not np.isfinite(points).all():
            self.data.clear()
            return None
        if self.data and (timestamp <= self.data[-1][0] or timestamp-self.data[-1][0] > self.max_gap):
            self.data.clear()
        self.data.append((timestamp, points.copy()))
        while len(self.data) > 1 and self.data[1][0] <= timestamp-self.seconds:
            self.data.popleft()
        if len(self.data) < 3 or timestamp-self.data[0][0] < self.seconds-1/self.fps-1e-6:
            return None
        # Interpolation is restricted to recorded history. A missing first sample
        # of up to one input interval is edge-held; it never creates new evidence.
        target = np.linspace(timestamp-self.seconds, timestamp, self.samples)
        times = np.asarray([t for t, _ in self.data])
        values = np.stack([p for _, p in self.data]).reshape(len(times), -1)
        output = np.stack([np.interp(target, times, values[:, k]) for k in range(51)], axis=-1)
        return output.reshape(self.samples, 17, 3).astype(np.float32)


class Alarm:
    def __init__(self, hold=.25, cooldown=5):
        self.hold, self.cooldown = hold, cooldown
        self.start, self.last, self.until = None, None, -float('inf')

    def update(self, timestamp, active, max_gap):
        if not active or (self.last is not None and timestamp-self.last > max_gap):
            self.start = None
        self.last = timestamp
        if not active:
            return False
        if self.start is None:
            self.start = timestamp
        if timestamp-self.start >= self.hold and timestamp >= self.until:
            self.until = timestamp+self.cooldown
            return True
        return False
