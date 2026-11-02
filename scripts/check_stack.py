#!/usr/bin/env python3
"""Read local private connection file; verify HTTP/RTSP without printing secrets."""
import json
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
lines = (root/"deploy/runtime/ha-connection.txt").read_text().splitlines()
rtsp = next(line.removeprefix("RTSP: ") for line in lines if line.startswith("RTSP: "))
snapshot = next(line.removeprefix("Snapshot: ") for line in lines if line.startswith("Snapshot: "))
state_url = snapshot.replace("/snapshot.jpg", "/state")
def check_video():
    for attempt in range(10):
        try:
            with urlopen(state_url, timeout=5) as response:
                state = json.load(response)
            break
        except URLError as exc:
            if (isinstance(exc, HTTPError) and exc.code != 503) or attempt == 9:
                raise
            time.sleep(1)
    print(json.dumps({"http": "ok", "count": state["count"], "postures": state["postures"]}))
    probe = None
    for attempt in range(3):
        try:
            probe = subprocess.run(["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-read_intervals", "%+2",
                                    "-show_entries", "stream=codec_name,width,height,r_frame_rate", "-of", "json", rtsp],
                                   capture_output=True, text=True, timeout=15)
            if probe.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            probe = None  # Never print an exception containing credential-bearing argv.
        time.sleep(1)
    if probe is None or probe.returncode:
        raise SystemExit("RTSP probe failed (credentials suppressed); check container logs")
    streams = json.loads(probe.stdout)["streams"]
    assert any(stream["codec_name"] == "h264" for stream in streams)
    print(json.dumps({"rtsp": "ok", "streams": streams}))


def check_mqtt():
    private = dict(line.split("=", 1) for line in (root/"deploy/.env").read_text().splitlines() if "=" in line)
    result = subprocess.run(["sudo", "-n", "docker", "run", "--rm", "--network", "indoorcam_default",
        "-e", "MQTT_TEST_HOST=mqtt", "-e", "MQTT_TEST_PORT=1883", "-e", "MQTT_TEST_PASSWORD="+private["MQTT_PASSWORD"],
        "-v", f"{root}:/work:ro", "-w", "/work", "--entrypoint", "python", "indoorcam:pn41",
        "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_mqtt_integration.py"])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    import sys
    if "--mqtt-only" not in sys.argv:
        check_video()
    check_mqtt()
