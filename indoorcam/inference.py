"""MobileDet / SSD MobileNet v2 (COCO person class 0) + per-person MoveNet."""
from pathlib import Path
import cv2
import numpy as np


def make_interpreter(path, device=None):
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        from tensorflow.lite import Interpreter
        from tensorflow.lite.experimental import load_delegate
    delegates = [load_delegate("libedgetpu.so.1", {"device": device})] if device else []
    interpreter = Interpreter(model_path=str(path), experimental_delegates=delegates, num_threads=2)
    interpreter.allocate_tensors()
    return interpreter


def put_image(interpreter, rgb):
    detail = interpreter.get_input_details()[0]
    _, h, w, _ = detail["shape"]
    image = cv2.resize(rgb, (int(w), int(h)))
    # Both shipped models consume unnormalized 0..255 RGB. MoveNet PTQ is uint8.
    image = image.astype(detail["dtype"])
    interpreter.set_tensor(detail["index"], image[None])


class Inference:
    def __init__(self, cfg):
        root = Path(cfg.model_dir)
        self.cfg = cfg
        model = "mobiledet" if cfg.detector_model == "mobiledet" else "person"
        self.detector = make_interpreter(root / f"{model}_{'edgetpu' if cfg.backend == 'coral' else 'cpu'}.tflite",
                                         cfg.coral_device if cfg.backend == "coral" else None)
        pose_backend = cfg.backend if cfg.pose_backend == "auto" else cfg.pose_backend
        pose_name = "pose_thunder" if cfg.pose_model == "thunder" else "pose"
        pose_file = pose_name + ("_edgetpu.tflite" if pose_backend == "coral" else ".tflite")
        self.pose = make_interpreter(root / pose_file,
                                     (cfg.pose_device or cfg.coral_device) if pose_backend == "coral" else None)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = bgr.shape[:2]
        crops = [(0, 0, width, height)]
        tiles = self.cfg.detector_tiles
        if tiles > 1:
            # Overlapping crops give small/distant people more detector pixels.
            cw, ch = round(width*1.3/tiles), round(height*1.3/tiles)
            crops += [(int(x), int(y), cw, ch) for y in np.linspace(0, height-ch, tiles)
                      for x in np.linspace(0, width-cw, tiles)]
        people = []
        for ox, oy, w, h in crops:
            put_image(self.detector, rgb[oy:oy+h, ox:ox+w])
            self.detector.invoke()
            boxes, classes, scores, count = [self.detector.get_tensor(d["index"]) for d in self.detector.get_output_details()]
            for i in range(int(count.flat[0])):
                score = float(scores[0, i])
                if int(classes[0, i]) != 0 or score < self.cfg.threshold:
                    continue
                y1, x1, y2, x2 = boxes[0, i]
                if tiles > 1 and (w != width or h != height):
                    # Avoid counting a partial person a second time where a tile
                    # cuts through a body. Overlap/full-frame pass cover its neighbors.
                    if ((ox > 0 and x1 < 0.03) or (oy > 0 and y1 < 0.03)
                        or (ox+w < width and x2 > 0.97) or (oy+h < height and y2 > 0.97)):
                        continue
                box = np.array([ox+x1*w, oy+y1*h, ox+x2*w, oy+y2*h]).clip([0, 0, 0, 0], [width-1, height-1, width-1, height-1])
                if box[2] - box[0] < 10 or box[3] - box[1] < max(10, height*self.cfg.min_person_height):
                    continue
                people.append({"bbox": box.tolist(), "score": score})
        if not people:
            return []
        xywh = [[p["bbox"][0], p["bbox"][1], p["bbox"][2]-p["bbox"][0], p["bbox"][3]-p["bbox"][1]] for p in people]
        indices = cv2.dnn.NMSBoxes(xywh, [p["score"] for p in people], self.cfg.threshold, 0.4)
        return [people[int(i)] for i in np.asarray(indices).reshape(-1)][:self.cfg.max_people]

    def verified(self, person):
        if not self.cfg.verify_person_pose or person["score"] >= 0.9:
            return True
        confidence = person["keypoints"][:, 2]
        # Weak SSD detections on chairs/blankets often have plausible boxes but
        # no shoulder evidence. Occluded real people can also fail this gate.
        return bool(max(confidence[5:7]) >= 0.25 and np.count_nonzero(confidence >= 0.3) >= 3)

    def keypoints(self, bgr, box):
        # A square padded crop preserves anatomy and includes limbs beyond SSD's box.
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = box
        side = max(x2-x1, y2-y1) * 1.25
        left, top = int((x1+x2-side)/2), int((y1+y2-side)/2)
        size = max(32, int(side))
        crop = np.zeros((size, size, 3), np.uint8)
        sx, sy, ex, ey = max(0, left), max(0, top), min(w, left+size), min(h, top+size)
        crop[sy-top:ey-top, sx-left:ex-left] = bgr[sy:ey, sx:ex]
        put_image(self.pose, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self.pose.invoke()
        detail = self.pose.get_output_details()[0]
        points = self.pose.get_tensor(detail["index"]).reshape(17, 3).astype(float)
        if np.issubdtype(detail["dtype"], np.integer):
            scale, zero = detail["quantization"]
            points = (points-zero)*scale
        # COCO order; convert [y,x,score] to full-image [x,y,score].
        return np.stack([points[:, 1]*size+left, points[:, 0]*size+top, points[:, 2]], axis=1)
