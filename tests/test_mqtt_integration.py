"""Opt-in test against a real authenticated Mosquitto; no fake MQTT transport."""
import json
import os
import queue
import time
import uuid
import paho.mqtt.client as mqtt
import pytest
from indoorcam.config import Config
from indoorcam.mqtt import HomeAssistant


@pytest.mark.skipif(not os.getenv("MQTT_TEST_HOST"), reason="Set MQTT_TEST_HOST/PORT/PASSWORD for a real broker")
def test_discovery_state_event_and_offline():
    ident = "it_"+uuid.uuid4().hex[:10]
    cfg = Config(camera_id=ident, mqtt_host=os.environ["MQTT_TEST_HOST"],
                 mqtt_port=int(os.getenv("MQTT_TEST_PORT", "18883")), mqtt_password_env="MQTT_TEST_PASSWORD")
    messages = queue.Queue()
    subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    subscriber.username_pw_set(cfg.mqtt_user, os.environ["MQTT_TEST_PASSWORD"])

    def connected(client, userdata, flags, reason, properties):
        client.subscribe([(f"homeassistant/+/{ident}/#", 1), (f"indoorcam/{ident}/#", 1)])
    subscriber.on_connect = connected
    subscriber.on_message = lambda c, u, msg: messages.put((msg.topic, msg.payload, msg.retain))
    subscriber.connect(cfg.mqtt_host, cfg.mqtt_port)
    subscriber.loop_start()
    bridge = None
    seen = {}

    def wait_for(predicate):
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            try:
                topic, body, retained = messages.get(timeout=0.2)
                seen[topic] = (body, retained)
            except queue.Empty:
                pass
            if predicate():
                return
        raise AssertionError(f"Missing MQTT condition; topics observed: {list(seen)}")

    base = f"indoorcam/{ident}"
    try:
        # Wait for the SUBACK path to settle before starting publisher/discovery.
        time.sleep(0.2)
        bridge = HomeAssistant(cfg)
        wait_for(lambda: f"homeassistant/event/{ident}/fall/config" in seen)
        state = {"count":1, "postures":{"standing":0,"sitting":0,"lying":1,"unknown":0},
                 "fall_candidate":True, "people":[],
                 "events":[{"type":"fall_candidate", "track_id":1, "time":1.0}]}
        bridge.update(state)
        wait_for(lambda: base+"/event" in seen and seen.get(base+"/availability", (None,))[0] == b"online")
        assert json.loads(seen[base+"/state"][0])["count"] == 1
        assert json.loads(seen[base+"/event"][0])["event_type"] == "fall_candidate"
        # A new subscriber must receive retained state/discovery, but no past event.
        late = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        received = []
        late.username_pw_set(cfg.mqtt_user, os.environ["MQTT_TEST_PASSWORD"])
        late.on_message = lambda c,u,m: received.append(m.topic)
        late.on_connect = lambda c,u,f,r,p: c.subscribe(base+"/#", qos=1)
        late.connect(cfg.mqtt_host, cfg.mqtt_port)
        late.loop_start()
        try:
            time.sleep(0.5)
            assert base+"/state" in received
            assert base+"/event" not in received
        finally:
            late.disconnect()
            late.loop_stop()
        bridge.offline()
        wait_for(lambda: seen.get(base+"/availability", (None,))[0] == b"offline")
    finally:
        if bridge:
            bridge.close()
        for topic in list(seen):
            subscriber.publish(topic, b"", qos=1, retain=True).wait_for_publish(timeout=2)
        subscriber.disconnect()
        subscriber.loop_stop()
