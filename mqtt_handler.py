"""
mqtt_handler.py — MQTT publisher with Shiftr.io support
========================================================
Supports both:
  - Local Mosquitto: set MQTT_USER = "" in config.py
  - Shiftr.io cloud: set MQTT_USER + MQTT_PASSWORD in config.py
Auto-reconnects if broker goes offline.
"""

import paho.mqtt.client as mqtt
import time
import threading
import config


class MQTTHandler:
    def __init__(self):
        self._client    = mqtt.Client()
        self._connected = False
        self._lock      = threading.Lock()

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Set credentials if configured (required for Shiftr.io)
        user = getattr(config, 'MQTT_USER',     '')
        pwd  = getattr(config, 'MQTT_PASSWORD', '')
        if user:
            self._client.username_pw_set(user, pwd)
            print(f"[MQTT] Using credentials for: {user}")
        else:
            print("[MQTT] No credentials — local broker mode")

        # Set reconnect delay ONCE at init (not inside loop)
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)

        t = threading.Thread(target=self._connect_loop, daemon=True)
        t.start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Connected to {config.MQTT_BROKER}:{config.MQTT_PORT}")
        else:
            self._connected = False
            codes = {
                1: "wrong protocol version",
                2: "invalid client ID",
                3: "broker unavailable",
                4: "wrong username/password — check MQTT_USER/MQTT_PASSWORD",
                5: "not authorized"
            }
            print(f"[MQTT] Connection failed: {codes.get(rc, f'rc={rc}')}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect — will retry")

    def _connect_loop(self):
        """
        Tries to connect once. After that, paho's loop_start()
        handles automatic reconnection — no manual loop needed.
        """
        self._client.loop_start()
        while not self._connected:
            try:
                self._client.connect(config.MQTT_BROKER,
                                     config.MQTT_PORT, 60)
                time.sleep(1.5)   # wait for on_connect callback
            except Exception as e:
                print(f"[MQTT] Cannot reach broker: {e} "
                      f"— retry in {config.MQTT_RECONNECT_DELAY}s")
                time.sleep(config.MQTT_RECONNECT_DELAY)

    def publish(self, device: str, action: str):
        if not self._connected:
            print(f"[MQTT] Not connected — dropped: {device}/{action}")
            return False
        topic = f"{config.MQTT_TOPIC_BASE}{device}/{action}"
        with self._lock:
            result = self._client.publish(topic, action, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] ✓  {topic}")
            return True
        print(f"[MQTT] ✗  publish failed (rc={result.rc})")
        return False

    def is_connected(self) -> bool:
        return self._connected

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
        print("[MQTT] Disconnected.")
