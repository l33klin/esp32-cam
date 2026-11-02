#!/usr/bin/env python3
"""Generate private runtime configs without editing HA or any existing broker.

Uses stdlib only. Run once; refuses to replace existing credentials.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets


def mosquitto_password(password):
    # Mosquitto password-file format $7$ = PBKDF2-HMAC-SHA512.
    salt = secrets.token_bytes(12)
    iterations = 101
    digest = hashlib.pbkdf2_hmac("sha512", password.encode(), salt, iterations)
    return f"$7${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lan-ip", required=True, help="pn41 LAN interface IP, e.g. 192.168.50.102")
    parser.add_argument("--source", default="/media/test.mkv")
    parser.add_argument("--apex-gid", default="1004")
    parser.add_argument("--loop", action="store_true", help="Loop a test file, never enable for a camera")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root/"deploy/runtime"
    envfile = root/"deploy/.env"
    if runtime.exists() or envfile.exists():
        raise SystemExit("Runtime credentials already exist. Edit them explicitly; refusing to overwrite.")
    runtime.mkdir(mode=0o700)
    (runtime/"media").mkdir()
    mqtt, publisher, viewer = [secrets.token_urlsafe(24) for _ in range(3)]
    files = {
        "passwords": "indoorcam:"+mosquitto_password(mqtt)+"\n",
        "mosquitto.conf": "listener 1883\nallow_anonymous false\npassword_file /mosquitto/config/passwords\npersistence true\npersistence_location /mosquitto/data/\nlog_dest stdout\n",
        "mediamtx.json": json.dumps({"rtspTransports": ["tcp"], "hls": False, "webrtc": False, "rtmp": False,
            "srt": False, "api": False, "authInternalUsers": [
                {"user": "publisher", "pass": publisher, "ips": [], "permissions": [{"action": "publish", "path": "annotated"}]},
                {"user": "viewer", "pass": viewer, "ips": [], "permissions": [{"action": "read", "path": "annotated"}]}],
            "paths": {"annotated": {"source": "publisher"}}}, indent=2),
        "config.json": json.dumps({"source": args.source, "backend": "coral", "coral_device": "pci:0",
            "pose_backend": "coral", "pose_device": "pci:1",
            "model_dir": "/app/models", "fps": 6, "width": 960, "height": 540,
            "roi": [], "roi_anchor": "bottom_center", "realtime": True, "loop": args.loop,
            "rtsp_output": f"rtsp://publisher:{publisher}@mediamtx:8554/annotated",
            "mqtt_host": "mqtt", "mqtt_user": "indoorcam", "mqtt_port": 1883,
            "api_host": "0.0.0.0", "api_port": 8080}, indent=2),
        "ha-connection.txt": f"MQTT host: {args.lan_ip}\nMQTT port: 18883\nMQTT user: indoorcam\nMQTT password: {mqtt}\n"
            f"RTSP: rtsp://viewer:{viewer}@{args.lan_ip}:18554/annotated\nSnapshot: http://{args.lan_ip}:18081/snapshot.jpg\n"
    }
    for name, content in files.items():
        path = runtime/name
        path.write_text(content)
        # Containers use different unprivileged UIDs; enclosing dir is private on host.
        path.chmod(0o644 if name not in ("ha-connection.txt",) else 0o600)
    envfile.write_text(f"LAN_IP={args.lan_ip}\nAPEX_GID={args.apex_gid}\nMQTT_PASSWORD={mqtt}\n")
    envfile.chmod(0o600)
    print(f"Created {runtime}. Credentials are in ha-connection.txt (not printed).")


if __name__ == "__main__":
    main()
