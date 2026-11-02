"""Geometry-based posture and conservative temporal fall candidate detection.

No learned fall classifier: outputs are heuristic candidates, never medical evidence.
All times are monotonic seconds (file mode uses media timestamps).
"""
from collections import deque
from dataclasses import dataclass, field
import math
import cv2
import numpy as np


def angle(a, b, c):
    u, v = a-b, c-b
    return math.degrees(math.acos(float(np.clip(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-8), -1, 1))))


def posture(points, threshold):
    def pair(a, b):
        good = [points[i, :2] for i in (a, b) if points[i, 2] >= threshold]
        return np.mean(good, axis=0) if good else None
    shoulders, hips = pair(5, 6), pair(11, 12)
    if shoulders is None or hips is None:
        return "unknown", None
    torso = hips - shoulders
    length = float(np.linalg.norm(torso))
    if length < 5:
        return "unknown", None
    tilt = math.degrees(math.atan2(abs(torso[0]), abs(torso[1])+1e-6))
    features = {"hip_y": float(hips[1]), "torso": length, "tilt": tilt}
    knees = []
    thighs = []
    horizontal_body = False
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if min(points[hip, 2], points[knee, 2]) >= threshold:
            d = points[knee, :2]-points[hip, :2]
            thighs.append(abs(d[1])/(np.linalg.norm(d)+1e-8))
            if points[ankle, 2] >= threshold:
                knees.append(angle(points[hip, :2], points[knee, :2], points[ankle, :2]))
                body = points[ankle, :2]-shoulders
                horizontal_body = horizontal_body or (abs(body[0]) > 1.4*abs(body[1]) and np.linalg.norm(body) > 1.3*length)
    # A bent torso alone is insufficient: kneeling/bending to pick up a child
    # must not be labeled lying merely because the shoulders are low.
    if tilt >= 55 and horizontal_body and knees and max(knees) > 145:
        return "lying", features
    if tilt < 45 and thighs:
        if min(thighs) < 0.65 or (knees and min(knees) < 135):
            return "sitting", features
        if max(thighs) > 0.75 and knees and max(knees) > 150:
            return "standing", features
    return "unknown", features


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    return inter / (max(0, a[2]-a[0])*max(0, a[3]-a[1])+max(0, b[2]-b[0])*max(0, b[3]-b[1])-inter+1e-8)


@dataclass
class Track:
    id: int
    bbox: list
    seen: float
    posture: str = "unknown"
    pending: str = "unknown"
    pending_since: float = 0.0
    history: deque = field(default_factory=deque)
    fall_since: object = None
    cooldown_until: float = 0.0
    alarm_until: float = 0.0

    def update(self, raw, features, now, cfg):
        if raw != self.pending:
            self.pending, self.pending_since = raw, now
        if now-self.pending_since >= cfg.posture_hold:
            self.posture = raw
        # Do not bridge tracking gaps with temporal fall evidence.
        if now-self.seen > min(cfg.track_timeout, 0.5):
            self.history.clear()
            self.fall_since = None
        while self.history and now-self.history[0][0] > cfg.fall_window:
            self.history.popleft()
        event = False
        if raw == "lying" and features and now >= self.cooldown_until:
            if self.fall_since is None:
                for _, old, f in self.history:
                    if old in ("standing", "sitting") and f:
                        if features["hip_y"]-f["hip_y"] > cfg.fall_drop*f["torso"]:
                            self.fall_since = now
                            break
            if self.fall_since is not None and now-self.fall_since >= cfg.fall_hold:
                event = True
                self.alarm_until = now+cfg.fall_latch
                self.cooldown_until = now+cfg.fall_cooldown
                self.fall_since = None
        else:
            self.fall_since = None
        self.history.append((now, raw, features))
        self.seen = now
        return event


class Analytics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()

    def inside(self, box, shape):
        if not self.cfg.roi:
            return True
        h, w = shape[:2]
        x = (box[0]+box[2])/2/w
        y = (box[1]+box[3])/2/h if self.cfg.roi_anchor == "center" else box[3]/h
        return cv2.pointPolygonTest(np.array(self.cfg.roi, np.float32), (x, y), False) >= 0

    def update(self, detections, now, shape):
        cfg = self.cfg
        self.tracks = {i: t for i, t in self.tracks.items() if now-t.seen <= cfg.track_timeout}
        available = set(self.tracks)
        # Global greedy association by IoU, with a normalized center-distance fallback.
        pairs = []
        for j, d in enumerate(detections):
            for i, t in self.tracks.items():
                b, a = np.array(d["bbox"]), np.array(t.bbox)
                overlap = iou(a, b)
                distance = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2) / max(20, np.linalg.norm(a[2:]-a[:2]))
                if overlap >= 0.15 or distance < 0.35:
                    pairs.append((overlap-distance*0.3, i, j))
        matches, used = {}, set()
        for _, i, j in sorted(pairs, reverse=True):
            if i in available and j not in used:
                matches[j] = i
                available.remove(i)
                used.add(j)
        people, events = [], []
        for j, d in enumerate(detections):
            if j not in matches:
                i = self.next_id
                self.next_id += 1
                self.tracks[i] = Track(i, d["bbox"], now)
                matches[j] = i
            t = self.tracks[matches[j]]
            raw, features = posture(d["keypoints"], cfg.keypoint_threshold)
            inside = self.inside(d["bbox"], shape)
            event = t.update(raw, features, now, cfg)
            t.bbox = d["bbox"]
            if not inside:
                # Evidence collected outside the monitored region cannot arm a
                # delayed alarm when the same track later enters the region.
                t.history.clear()
                t.fall_since = None
                t.alarm_until = 0
            person = {"id": t.id, "bbox": t.bbox, "score": d["score"], "posture": t.posture,
                      "raw_posture": raw, "in_roi": inside,
                      "fall_candidate": bool(inside and now < t.alarm_until),
                      "keypoints": d["keypoints"].tolist()}
            people.append(person)
            if event and inside:
                events.append({"type": "fall_candidate", "track_id": t.id, "time": now,
                               "posture": raw, "bbox": t.bbox})
        counts = {p: sum(x["in_roi"] and x["posture"] == p for x in people)
                  for p in ("standing", "sitting", "lying", "unknown")}
        return {"time": now, "count": sum(counts.values()), "postures": counts, "people": people,
                "fall_candidate": any(p["fall_candidate"] for p in people), "events": events,
                "truncated": len(detections) >= cfg.max_people}
