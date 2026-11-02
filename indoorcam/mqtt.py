import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

LOG = logging.getLogger(__name__)


class HomeAssistant:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = f"{cfg.topic_prefix}/{cfg.camera_id}"
        self.fresh_until = 0
        self.last_state = None
        self.stop = threading.Event()
        self.online = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{cfg.camera_id}_analytics")
        if cfg.mqtt_user:
            password = os.getenv(cfg.mqtt_password_env)
            if not password:
                raise ValueError(f"Missing MQTT password environment variable {cfg.mqtt_password_env}")
            self.client.username_pw_set(cfg.mqtt_user, password)
        self.client.will_set(f"{self.base}/availability", "offline", qos=1, retain=True)
        self.client.on_connect = self._connect
        self.client.on_message = self._message
        self.client.reconnect_delay_set(1, 30)
        self.client.max_queued_messages_set(64)
        self.client.connect_async(cfg.mqtt_host, cfg.mqtt_port, keepalive=15)
        self.client.loop_start()
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _watch(self):
        while not self.stop.wait(0.5):
            if self.online and time.monotonic() >= self.fresh_until:
                self.offline()

    def _send(self, topic, data, retain=True):
        return self.client.publish(topic, json.dumps(data) if not isinstance(data, str) else data, qos=1, retain=retain)

    def _connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            LOG.error("MQTT rejected connection: %s", reason_code)
            return
        client.subscribe(f"{self.cfg.mqtt_discovery_prefix}/status", qos=1)
        self.discovery()
        if self.last_state is not None and time.monotonic() < self.fresh_until:
            self._send(f"{self.base}/state", self.last_state)
            self._send(f"{self.base}/availability", "online")
            self.online = True
        else:
            self._send(f"{self.base}/availability", "offline")
            self.online = False

    def _message(self, client, userdata, msg):
        if msg.payload == b"online":
            self.discovery()

    def discovery(self):
        c = self.cfg
        common = {"device": {"identifiers": [c.camera_id], "name": "Indoor Camera", "manufacturer": "DIY ESP32 + Coral",
                              "model": f"{c.detector_model} + MoveNet"},
                  "availability_topic": f"{self.base}/availability", "state_topic": f"{self.base}/state"}
        entities = [("sensor", "count", "People", "{{ value_json.count }}", None)]
        entities += [("sensor", p, p.title(), "{{ value_json.postures."+p+" }}", None)
                     for p in ("standing", "sitting", "lying", "unknown")]
        entities += [("binary_sensor", "occupied", "Occupied", "{{ 'ON' if value_json.count > 0 else 'OFF' }}", "occupancy"),
                     ("binary_sensor", "fall_candidate", "Possible fall", "{{ 'ON' if value_json.fall_candidate else 'OFF' }}", "problem")]
        for domain, key, name, template, device_class in entities:
            data = {**common, "unique_id": f"{c.camera_id}_{key}", "name": name, "value_template": template}
            if device_class:
                data["device_class"] = device_class
            self._send(f"{c.mqtt_discovery_prefix}/{domain}/{c.camera_id}/{key}/config", data)
        self._send(f"{c.mqtt_discovery_prefix}/event/{c.camera_id}/fall/config",
                   {**common, "unique_id": f"{c.camera_id}_fall_event", "name": "Fall candidate event",
                    "state_topic": f"{self.base}/event", "event_types": ["fall_candidate"], "device_class": "motion"})

    def update(self, state):
        # Exclude bulky skeletons and transient events from retained state.
        self.last_state = {k: v for k, v in state.items() if k not in ("people", "events")}
        self.fresh_until = time.monotonic()+5
        if not self.client.is_connected():
            return
        self._send(f"{self.base}/state", self.last_state)
        if not self.online:
            self._send(f"{self.base}/availability", "online")
            self.online = True
        for event in state["events"]:
            self._send(f"{self.base}/event", {**event, "event_type": event["type"],
                       "timestamp": datetime.now(timezone.utc).isoformat()}, retain=False)

    def offline(self):
        self.fresh_until = 0
        if self.online and self.client.is_connected():
            self._send(f"{self.base}/availability", "offline")
        self.online = False

    def close(self):
        self.stop.set()
        self.watchdog.join(timeout=1)
        result = self._send(f"{self.base}/availability", "offline")
        if self.client.is_connected():
            result.wait_for_publish(timeout=2)
        self.client.disconnect()
        self.client.loop_stop()
