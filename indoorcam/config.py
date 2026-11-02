from dataclasses import dataclass, field, fields
from pathlib import Path
import os
import yaml


@dataclass
class Config:
    source: str = ""
    backend: str = "coral"
    coral_device: str = "pci:0"
    pose_backend: str = "auto"
    pose_device: str = ""
    pose_model: str = "lightning"
    model_dir: str = "models"
    fps: float = 8.0
    width: int = 960
    height: int = 540
    threshold: float = 0.45
    keypoint_threshold: float = 0.3
    max_people: int = 10
    detector_tiles: int = 2
    detector_model: str = "mobiledet"
    verify_person_pose: bool = True
    min_person_height: float = 0.06
    roi: list = field(default_factory=list)
    roi_anchor: str = "bottom_center"
    posture_hold: float = 0.5
    track_timeout: float = 0.8
    fall_window: float = 1.2
    fall_drop: float = 0.25
    fall_hold: float = 1.5
    fall_cooldown: float = 30.0
    fall_latch: float = 10.0
    rtsp_output: str = ""
    output_file: str = ""
    jsonl: str = ""
    loop: bool = False
    realtime: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 0
    camera_id: str = "indoor_cam"
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_user: str = "indoorcam"
    mqtt_password_env: str = "MQTT_PASSWORD"
    mqtt_discovery_prefix: str = "homeassistant"
    topic_prefix: str = "indoorcam"

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        obj = cls(**data)
        # Only explicit substitutions; do not expand arbitrary config contents.
        obj.source = os.getenv("CAMERA_SOURCE", obj.source)
        obj.validate()
        return obj

    def validate(self):
        if not self.source:
            raise ValueError("source / CAMERA_SOURCE is required")
        if self.backend not in ("coral", "cpu"):
            raise ValueError("backend must be coral or cpu; no silent CPU fallback")
        if self.pose_backend not in ("auto", "coral", "cpu"):
            raise ValueError("pose_backend must be auto, coral, or cpu")
        if self.pose_model not in ("lightning", "thunder"):
            raise ValueError("pose_model must be lightning or thunder")
        if self.fps <= 0 or self.width < 32 or self.height < 32 or self.width % 2 or self.height % 2:
            raise ValueError("fps must be positive; dimensions must be positive even numbers >= 32")
        if not 0 < self.threshold < 1 or not 0 < self.keypoint_threshold < 1:
            raise ValueError("confidence thresholds must be between 0 and 1")
        if self.max_people < 1 or self.roi_anchor not in ("center", "bottom_center"):
            raise ValueError("Invalid max_people / roi_anchor")
        if self.detector_tiles not in (1, 2, 3):
            raise ValueError("detector_tiles must be 1 (full frame), 2, or 3")
        if self.detector_model not in ("mobilenet", "mobiledet"):
            raise ValueError("detector_model must be mobilenet or mobiledet")
        if not 0 <= self.min_person_height < 1:
            raise ValueError("min_person_height must be normalized in [0,1)")
        if self.roi and (len(self.roi) < 3 or any(len(p) != 2 or any(not 0 <= v <= 1 for v in p) for p in self.roi)):
            raise ValueError("roi needs >= 3 normalized [x,y] points in [0,1]")
        for name in ("posture_hold", "track_timeout", "fall_window", "fall_drop", "fall_hold", "fall_cooldown", "fall_latch"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

