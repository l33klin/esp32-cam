#!/usr/bin/env python3
"""Emit one explicitly tagged, non-retained integration-test event to HA.

Run inside the project's container/network with MQTT_PASSWORD in its environment.
The example HA automation ignores events with test=true.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import paho.mqtt.client as mqtt


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/"deploy/runtime/config.json").read_text())
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="indoorcam_ha_integration_check")
    client.username_pw_set(cfg.get("mqtt_user", "indoorcam"), os.environ["MQTT_PASSWORD"])
    ready = threading.Event()
    client.on_connect = lambda c,u,f,r,p: ready.set() if not r.is_failure else None
    client.connect(cfg["mqtt_host"], cfg.get("mqtt_port", 1883))
    client.loop_start()
    try:
        if not ready.wait(8):
            raise RuntimeError("MQTT connection did not become ready")
        payload = {"event_type": "fall_candidate", "type": "fall_candidate", "test": True,
                   "source": "ha_integration_check", "track_id": 0,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "message": "HA integration test; not a detected fall"}
        topic = f"{cfg.get('topic_prefix','indoorcam')}/{cfg.get('camera_id','indoor_cam')}/event"
        client.publish(topic, json.dumps(payload), qos=1, retain=False).wait_for_publish(timeout=5)
        print(json.dumps({"topic": topic, "test": True, "retained": False, "timestamp": payload["timestamp"]}))
    finally:
        client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    main()
