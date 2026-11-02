import cv2
import json
import os
import select
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def fit(frame, width, height):
    h, w = frame.shape[:2]
    scale = min(width/w, height/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    result = np.zeros((height, width, 3), np.uint8)
    x, y = (width-rw)//2, (height-rh)//2
    result[y:y+rh, x:x+rw] = cv2.resize(frame, (rw, rh))
    return result


def annotate(frame, state, cfg):
    frame = frame.copy()
    if cfg.roi:
        polygon = (np.array(cfg.roi)*[cfg.width, cfg.height]).astype(np.int32)
        cv2.polylines(frame, [polygon], True, (255, 200, 0), 2)
    edges = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    for person in state["people"]:
        color = (0, 0, 255) if person["fall_candidate"] else ((40, 220, 40) if person["in_roi"] else (150, 150, 150))
        x1, y1, x2, y2 = map(round, person["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{person['id']} {person['posture']} {person['score']:.2f}"
        if person["fall_candidate"]:
            label += " POSSIBLE FALL"
        cv2.putText(frame, label, (x1, max(45, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        points = person["keypoints"]
        for a, b in edges:
            if min(points[a][2], points[b][2]) >= cfg.keypoint_threshold:
                cv2.line(frame, tuple(map(round, points[a][:2])), tuple(map(round, points[b][:2])), color, 1)
    cv2.rectangle(frame, (0, 0), (cfg.width, 30), (20, 20, 20), -1)
    cv2.putText(frame, f"People in ROI: {state['count']}  " + "  ".join(f"{k}:{v}" for k, v in state["postures"].items()),
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class LatestCapture:
    """Drain live input on its own thread; never build an inference backlog."""
    def __init__(self, source, loop=False, rtsp_size=None):
        self.source, self.loop = source, loop
        self.rtsp_size = rtsp_size
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.serial = 0
        self.generation = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        if urlsplit(self.source).scheme == "rtsp" and self.rtsp_size:
            self._run_rtsp()
            return
        is_file = not urlsplit(self.source).scheme
        while not self.stop.is_set():
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            epoch, index = time.monotonic(), 0
            with self.lock:
                self.generation += 1
            while cap.isOpened() and not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if is_file and self.stop.wait(max(0, epoch+index/fps-time.monotonic())):
                    break
                with self.lock:
                    self.serial += 1
                    self.latest = (self.serial, time.monotonic(), frame, self.generation)
                index += 1
            cap.release()
            if not (is_file and self.loop):
                with self.lock:
                    self.latest = None
            if is_file and not self.loop:
                return
            self.stop.wait(0 if is_file and self.loop else 1)

    def _run_rtsp(self):
        """Use FFmpeg for sparse MJPEG RTSP without waiting for FPS estimation.

        OpenCV's bundled FFmpeg timed out probing the real OV3660 stream. A
        separate demuxer with a small probe starts on the first complete JPEG.
        Read raw frames without duplicating them to the configured output FPS.
        """
        width, height = self.rtsp_size
        frame_size = width*height*3
        while not self.stop.is_set():
            proc = None
            try:
                proc = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-probesize", "32", "-analyzeduration", "1", "-rtsp_transport", "tcp",
                    "-threads", "1", "-i", self.source, "-an", "-sn", "-dn",
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                    "-vsync", "0", "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                os.set_blocking(proc.stdout.fileno(), False)
                with self.lock:
                    self.generation += 1
                first = True
                while not self.stop.is_set():
                    raw = read_pipe_frame(proc.stdout.fileno(), frame_size, self.stop, 15 if first else 10)
                    if raw is None:
                        break
                    first = False
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    with self.lock:
                        self.serial += 1
                        self.latest = (self.serial, time.monotonic(), frame, self.generation)
            except OSError:
                pass
            finally:
                with self.lock:
                    self.latest = None
                if proc:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    proc.stdout.close()
            self.stop.wait(1)

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)


def read_pipe_frame(fd, size, stop, timeout):
    """Return one complete raw frame, or None on EOF, deadline, or shutdown."""
    raw = bytearray()
    deadline = time.monotonic()+timeout
    while len(raw) < size and not stop.is_set():
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            return None
        if select.select([fd], [], [], min(0.2, remaining))[0]:
            try:
                chunk = os.read(fd, min(65536, size-len(raw)))
            except BlockingIOError:
                continue
            if not chunk:
                return None
            raw.extend(chunk)
    return raw if len(raw) == size else None


class FrameHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = self.state = None
        self.updated = 0

    def set(self, frame, state):
        with self.lock:
            self.frame, self.state, self.updated = frame, state, time.monotonic()

    def get(self):
        with self.lock:
            return (self.frame, self.state) if time.monotonic()-self.updated < 5 else (None, None)

    def invalidate(self):
        with self.lock:
            self.updated = 0


def start_api(hub, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            frame, state = hub.get()
            if self.path not in ("/health", "/state", "/snapshot.jpg"):
                self.send_error(404)
                return
            good = frame is not None
            if self.path == "/snapshot.jpg" and good:
                body = cv2.imencode(".jpg", frame)[1].tobytes()
                content_type = "image/jpeg"
            else:
                body = json.dumps(state if good else {"status": "offline"}).encode()
                content_type = "application/json"
            self.send_response(200 if good else 503)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class RtspWriter:
    """Constant-rate H.264 publisher, independently paced from inference.

    Restarts ffmpeg after disconnect; bounds pipe writes so an unavailable RTSP
    endpoint cannot block inference or MQTT. Stops publication for stale sources.
    """
    def __init__(self, cfg, hub):
        self.cfg, self.hub = cfg, hub
        self.stop = threading.Event()
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _close_proc(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc.stdin.close()
            self.proc = None

    def _run(self):
        cfg = self.cfg
        deadline = time.monotonic()
        while not self.stop.is_set():
            frame, _ = self.hub.get()
            if frame is None:
                self._close_proc()
                self.stop.wait(0.2)
                deadline = time.monotonic()
                continue
            try:
                if self.proc is None:
                    self.proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                        "-pix_fmt", "bgr24", "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
                        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
                        "-g", str(max(1, int(cfg.fps*2))), "-bf", "0", "-f", "rtsp", "-rtsp_transport", "tcp", cfg.rtsp_output],
                        stdin=subprocess.PIPE)
                    os.set_blocking(self.proc.stdin.fileno(), False)
                view = memoryview(frame.tobytes())
                end = time.monotonic()+2
                while view and not self.stop.is_set():
                    if time.monotonic() > end:
                        raise TimeoutError("RTSP pipe stalled")
                    fd = self.proc.stdin.fileno()
                    if select.select([], [fd], [], 0.1)[1]:
                        try:
                            view = view[os.write(fd, view):]
                        except BlockingIOError:
                            pass
                deadline += 1/cfg.fps
                self.stop.wait(max(0, deadline-time.monotonic()))
                if time.monotonic()-deadline > 1:
                    deadline = time.monotonic()
            except (OSError, TimeoutError):
                self._close_proc()
                self.stop.wait(2)
                deadline = time.monotonic()
        self._close_proc()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
